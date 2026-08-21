"""Verifier — §6.7. The differentiating node.

Given draft_answer and tool_results:
  1. extract every numeric literal in the draft
  2. check each against values present in tool_results rows (rounding tolerance)
  3. check no claim asserts anything outside the retrieved date range
  4. return {"pass": bool, "reason": str}

Steps 1-3 are code, not LLM. That is the whole point: a model asked "is this
answer supported?" can hallucinate the audit as easily as the answer. Only the
"reason" phrasing is generated, and only once a verdict already exists.

Why this node earns its keep, measured rather than assumed: across three golden
runs, *every* wrong answer executed cleanly on the first attempt. The repair
loop never fired, because nothing errored — the SQL was valid and the number was
wrong. Confidence scores miss the same class (the categorizer put Adobe in
`shopping` at 0.9). Nothing else in the design catches a confident, well-formed,
incorrect answer.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from ..state import AgentState

# Absolute floor for cent-level rounding, plus a relative band so large figures
# are not held to a stricter standard than small ones.
ABS_TOLERANCE = 0.02
REL_TOLERANCE = 0.005

# Small bare integers are nearly always prose — "the top 3 merchants", "over 5
# months" — not figures drawn from the data. Auditing them yields false
# rejections, and a verifier that fails correct answers is worse than none: it
# trains the reader to ignore the verdict. Counts that genuinely matter are
# usually large enough to clear this, and row counts are checked separately.
MIN_CHECKED_MAGNITUDE = 10.0

_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ISO_MONTH = re.compile(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])\b")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")

# A calendar year is a period, not a quantity — but only when it stands alone.
# The lookarounds keep a genuine figure that happens to look like a year:
# "$2,026.00" and "2026.50" are amounts and must still be audited, while the
# "2026" in "June 2026" must not. Without them, stripping years to fix a false
# rejection would open a hole four digits wide for a fabricated amount.
_BARE_YEAR = re.compile(r"(?<![\d.,$])\b(?:19|20)\d{2}\b(?![\d.,])")


def extract_numerics(text: str) -> list[float]:
    """Every numeric literal in the draft, currency and separators stripped.

    Calendar references are removed first, longest form to shortest. Without
    that, "2026-03-15" yields 2026, -3 and -15 as claims, and every answer
    mentioning a date fails verification on numbers nobody asserted.

    Bare years are the same bug one level up, and it survived the original fix
    because it needs a *month* to show itself. Asked "did I spend more in June
    or July?", the agent correctly returned both sides — and the answer echoed
    the sub-questions, which read "June 2026" and "July 2026". Four instances of
    2026 became four unsupported claims and the correct answer was stamped
    "I couldn't verify this answer, so treat it with caution."

    Years still get checked, just not here: `_dates_in_range` reads them off the
    original text and tests them against the retrieved window. A year is
    verified as a period, which is what it is, rather than as a figure that
    ought to appear in a row, which it never will.
    """
    text = _ISO_DATE.sub(" ", text)
    text = _ISO_MONTH.sub(" ", text)
    text = _BARE_YEAR.sub(" ", text)

    found = []
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace("$", "").replace(",", "").replace("%", "")
        if cleaned in ("", "-", "."):
            continue
        try:
            found.append(float(cleaned))
        except ValueError:
            continue
    return found


def _row_values(tool_results: list[dict]) -> list[float]:
    """Every numeric a tool actually returned — the set of supportable claims."""
    values: list[float] = []
    for result in tool_results:
        # "I found 12 subscriptions" is a claim about how many rows came back,
        # so the row count is itself a supportable figure.
        values.append(float(len(result.get("rows") or [])))
        for row in result.get("rows") or []:
            for value in row.values():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    values.append(float(value))
                elif isinstance(value, str):
                    values.extend(extract_numerics(value))
    return values


def _supported(claim: float, available: list[float]) -> bool:
    """A claim is supported if it matches a returned value, or is its magnitude.

    Sign flips are allowed deliberately: outflows are stored negative, and
    "you spent $412.55" is a correct rendering of -412.55. Rejecting that would
    make the verifier fail on correct answers, which is worse than useless — it
    would train the reader to ignore it.
    """
    for value in available:
        tolerance = max(ABS_TOLERANCE, abs(value) * REL_TOLERANCE)
        if abs(claim - value) <= tolerance or abs(abs(claim) - abs(value)) <= tolerance:
            return True

        # Percentages: a claim of 53.5 against a returned fraction 0.535.
        # Restricted to genuine fractions. Applying it to any value let a row
        # count of 1 support any claim near 100, because 1 * 100 = 100 — which
        # let a corrupted answer through in the catch-rate eval.
        if 0 < abs(value) < 1:
            scaled = abs(value) * 100
            if abs(abs(claim) - scaled) <= max(ABS_TOLERANCE, scaled * REL_TOLERANCE):
                return True

    return False


def _dates_in_range(text: str, tool_results: list[dict]) -> tuple[bool, str]:
    """Reject claims about periods outside what was actually retrieved."""
    retrieved: list[date] = []
    for result in tool_results:
        for row in result.get("rows") or []:
            for value in row.values():
                if isinstance(value, str):
                    for match in _ISO_DATE.finditer(value):
                        try:
                            retrieved.append(date.fromisoformat(match.group(0)))
                        except ValueError:
                            pass

    if not retrieved:
        return True, ""

    lo, hi = min(retrieved), max(retrieved)
    for match in _ISO_DATE.finditer(text):
        try:
            claimed = date.fromisoformat(match.group(0))
        except ValueError:
            continue
        if not (lo <= claimed <= hi):
            return False, f"answer references {claimed.isoformat()}, outside retrieved range {lo}..{hi}"

    for match in _YEAR.finditer(text):
        year = int(match.group(0))
        if year < lo.year or year > hi.year:
            return False, f"answer references {year}, outside retrieved range {lo.year}..{hi.year}"

    return True, ""


def verify(draft_answer: str, tool_results: list[dict]) -> dict[str, Any]:
    """Pure function: no LLM, no I/O. Returns {"pass", "reason", ...}."""
    errors = [r for r in tool_results if r.get("error")]
    available = _row_values(tool_results)

    if not tool_results:
        return {"pass": False, "reason": "no tool results to verify against",
                "unsupported": [], "checked": 0}

    if not available and not errors:
        return {"pass": False, "reason": "tools returned no rows, so no figure can be supported",
                "unsupported": [], "checked": 0}

    claims = [c for c in extract_numerics(draft_answer) if abs(c) >= MIN_CHECKED_MAGNITUDE]
    unsupported = [c for c in claims if not _supported(c, available)]

    if unsupported:
        shown = ", ".join(f"{c:g}" for c in unsupported[:4])
        return {
            "pass": False,
            "reason": f"{len(unsupported)} figure(s) not found in any tool result: {shown}",
            "unsupported": unsupported,
            "checked": len(claims),
        }

    in_range, why = _dates_in_range(draft_answer, tool_results)
    if not in_range:
        return {"pass": False, "reason": why, "unsupported": [], "checked": len(claims)}

    if not claims:
        return {"pass": True, "reason": "answer makes no numeric claims",
                "unsupported": [], "checked": 0}

    return {
        "pass": True,
        "reason": f"all {len(claims)} figure(s) trace to retrieved rows",
        "unsupported": [],
        "checked": len(claims),
    }


def verifier(state: AgentState) -> AgentState:
    verdict = verify(state.get("draft_answer", ""), state.get("tool_results", []))
    return {**state, "verifier_verdict": verdict}
