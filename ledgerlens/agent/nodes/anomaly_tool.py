"""Statistical anomaly detection — §6.6.

Per category, compute a rolling baseline from prior months and flag deviations.
Returns structured findings, never prose: the answer node renders them and the
verifier reconciles the figures, so this node's job ends at the numbers.

Two decisions worth stating:

**The baseline excludes the month being judged.** Including it drags the mean
and inflates the standard deviation toward the outlier, so a large spike partly
hides itself — the bigger the anomaly, the more it suppresses its own z-score.
Measured on the synthetic travel spike: leave-one-out gives z=16.2, including
the month gives z=3.1. Both clear a threshold of 2, but the second understates
by 5x, and a smaller spike would vanish entirely.

**IQR runs alongside z-score.** A z-score needs a roughly symmetric baseline,
and monthly category spend often is not — one big month in an otherwise quiet
category skews the standard deviation enough to mask later anomalies. IQR is
rank-based and does not care. A month is flagged when either test fires, and the
finding says which, so a reader can tell a distribution problem from a real one.
"""

from __future__ import annotations

import sqlite3
import statistics as st
from dataclasses import asdict, dataclass

from ...db import connect_readonly
from ..state import AgentState

Z_THRESHOLD = 2.0
IQR_MULTIPLIER = 1.5
BASELINE_MONTHS = 6
MIN_BASELINE_MONTHS = 3

# Statistical significance is not the same as being worth telling someone about.
# A near-constant category collapses its own IQR fences, so any wobble looks
# extreme: subscriptions moved 50.77 -> 52.27 and scored z=2.24, and one month
# scored z=0.00 while still tripping IQR. Both are $1.50 of noise.
#
# A finding must clear BOTH gates, so neither a large percentage of a tiny
# number nor a tiny percentage of a large one gets through. Genuine subscription
# price changes are caught by §7's dedicated price-hike detector against
# recurring_series, which is the right tool for a $1.50 move.
MIN_RELATIVE_DEVIATION = 0.15   # 15% off baseline
MIN_ABSOLUTE_DEVIATION = 25.0   # and at least this many currency units


@dataclass
class Finding:
    category: str
    period: str            # YYYY-MM
    observed: float        # outflow magnitude, positive
    baseline_mean: float
    baseline_months: int
    z_score: float | None
    direction: str         # "above" | "below"
    method: str            # "z-score" | "iqr" | "z-score+iqr"

    @property
    def severity(self) -> float:
        """Ranking key, defined even when z-score is not.

        A perfectly stable baseline has zero variance, so z is undefined — yet
        a departure from it is the *most* unexpected thing that can happen, not
        the least. Ranking on `abs(z or 0)` sorted such findings last.
        """
        if self.z_score is not None:
            return abs(self.z_score)
        if not self.baseline_mean:
            return float("inf")
        return abs(self.observed - self.baseline_mean) / self.baseline_mean * 100

    def as_dict(self) -> dict:
        return asdict(self)


def _monthly_totals(conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
    """{category: {YYYY-MM: outflow magnitude}} over purchases only (§4)."""
    totals: dict[str, dict[str, float]] = {}
    for category, period, amount in conn.execute(
        """SELECT c.name, substr(t.posted_date,1,7), SUM(t.amount)
           FROM transactions t JOIN categories c ON c.id = t.category_id
           WHERE t.type = 'purchase'
           GROUP BY c.name, substr(t.posted_date,1,7)"""
    ):
        totals.setdefault(category, {})[period] = abs(amount)
    return totals


def _iqr_bounds(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    mid = len(ordered) // 2
    lower = ordered[:mid]
    upper = ordered[mid + 1:] if len(ordered) % 2 else ordered[mid:]
    q1, q3 = st.median(lower), st.median(upper)
    spread = q3 - q1
    return q1 - IQR_MULTIPLIER * spread, q3 + IQR_MULTIPLIER * spread


def _is_material(observed: float, baseline_mean: float) -> bool:
    """Big enough, both proportionally and in absolute terms, to be worth saying."""
    delta = abs(observed - baseline_mean)
    if delta < MIN_ABSOLUTE_DEVIATION:
        return False
    return baseline_mean > 0 and delta / baseline_mean >= MIN_RELATIVE_DEVIATION


def _assess(observed: float, baseline: list[float]) -> tuple[float | None, list[str]]:
    """Returns (z_score, methods that fired) for one month against its baseline."""
    methods: list[str] = []

    mean = st.mean(baseline)
    stdev = st.pstdev(baseline)
    z = (observed - mean) / stdev if stdev else None
    if z is not None and abs(z) > Z_THRESHOLD:
        methods.append("z-score")

    if len(baseline) >= 4:
        low, high = _iqr_bounds(baseline)
        if observed < low or observed > high:
            methods.append("iqr")

    return z, methods


def detect(conn: sqlite3.Connection, category: str | None = None) -> list[Finding]:
    """Flag months whose category spend deviates from the trailing baseline."""
    findings: list[Finding] = []

    for name, by_period in _monthly_totals(conn).items():
        if category and name != category:
            continue

        periods = sorted(by_period)
        for index, period in enumerate(periods):
            window = periods[max(0, index - BASELINE_MONTHS):index]
            # Trailing window only; a month is never part of its own baseline.
            baseline = [by_period[p] for p in window]
            if len(baseline) < MIN_BASELINE_MONTHS:
                continue

            observed = by_period[period]
            z, methods = _assess(observed, baseline)
            if not methods:
                continue
            if not _is_material(observed, st.mean(baseline)):
                continue

            findings.append(Finding(
                category=name,
                period=period,
                observed=round(observed, 2),
                baseline_mean=round(st.mean(baseline), 2),
                baseline_months=len(baseline),
                z_score=round(z, 2) if z is not None else None,
                direction="above" if observed > st.mean(baseline) else "below",
                method="+".join(methods),
            ))

    findings.sort(key=lambda f: f.severity, reverse=True)
    return findings


def anomaly_tool(state: AgentState) -> AgentState:
    """Returns structured findings as rows the verifier can reconcile against."""
    results = list(state.get("tool_results", []))

    targets = [s for s in state.get("plan", []) if s.get("tool") == "anomaly"]
    if not targets:
        targets = [{"sub_question": state["question"]}]

    for step in targets:
        question = step.get("sub_question", state["question"])
        try:
            with connect_readonly() as conn:
                findings = detect(conn)
            entry = {
                "tool": "anomaly",
                "sub_question": question,
                "query": "rolling baseline, z-score + IQR per category",
                "rows": [f.as_dict() for f in findings],
                "error": None,
            }
        except Exception as exc:
            entry = {
                "tool": "anomaly",
                "sub_question": question,
                "query": "",
                "rows": [],
                "error": f"anomaly detection failed: {exc}",
            }
        results.append(entry)

    return {**state, "tool_results": results}
