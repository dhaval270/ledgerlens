"""Periodicity detection — §5.5. Pure algorithm, no LLM.

Group by merchant, sort dates, compute gaps. A series needs >=3 occurrences and
a gap standard deviation under ~20% of the mean. Store cadence and typical
amount; a last_amount more than 5% above typical is the subscription price-hike
signal §7 consumes.

Two decisions worth stating, because both are load-bearing:

**typical_amount is the median, not the mean.** A price hike pulls the mean
toward itself, shrinking the deviation the detector is looking for. Measured on
the synthetic set (12 payments, 15% rise over the last two): median gives a
ratio of 1.150, mean gives 1.122 — both clear the 1.05 threshold, so the mean
would not have missed this one. The median matters at the margin, where a
smaller hike or a longer run of raised payments erodes the mean's signal while
leaving the median's intact.

**Amount stability is required alongside cadence.** The spec describes cadence
only, but frequent everyday spend (coffee runs, groceries) can drift into a
regular-looking rhythm across a year. Requiring a stable amount is what
separates "a subscription" from "I buy coffee most Tuesdays", and a false
recurring series would emit bogus price-hike insights every month.
"""

from __future__ import annotations

import sqlite3
import statistics as st
from dataclasses import dataclass
from datetime import date, timedelta

MIN_OCCURRENCES = 3
MAX_GAP_CV = 0.20        # stdev / mean of the day gaps
MAX_AMOUNT_CV = 0.15     # stdev / mean of the absolute amounts
PRICE_HIKE_RATIO = 1.05

# Only these types can form a series. Refunds are inflow noise against an
# otherwise regular merchant, and would corrupt both cadence and amount.
SERIES_TYPES = ("purchase", "income", "transfer")

# A series counts as active if it was seen within this many cadences of the most
# recent transaction in the ledger.
ACTIVE_CADENCE_SLACK = 1.8


@dataclass
class Series:
    merchant_id: int
    cadence_days: int
    typical_amount: float
    last_amount: float
    last_seen: str
    active: bool
    occurrences: int

    @property
    def is_price_hike(self) -> bool:
        if not self.typical_amount:
            return False
        return abs(self.last_amount) > abs(self.typical_amount) * PRICE_HIKE_RATIO


def _cv(values: list[float]) -> float:
    """Coefficient of variation. Zero spread is perfectly regular, not undefined."""
    mean = st.mean(values)
    if mean == 0:
        return float("inf")
    if len(values) < 2:
        return 0.0
    return st.stdev(values) / abs(mean)


def _analyze(rows: list[tuple[str, float]], ledger_end: date) -> Series | None:
    """rows: [(posted_date, amount)] for one merchant, ascending by date."""
    if len(rows) < MIN_OCCURRENCES:
        return None

    dates = [date.fromisoformat(d) for d, _ in rows]
    amounts = [a for _, a in rows]

    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    if not gaps or any(g <= 0 for g in gaps):
        return None  # same-day repeats are duplicates, not a cadence
    if _cv(gaps) > MAX_GAP_CV:
        return None
    if _cv([abs(a) for a in amounts]) > MAX_AMOUNT_CV:
        return None

    cadence = round(st.mean(gaps))
    last_seen = dates[-1]
    return Series(
        merchant_id=0,  # filled in by the caller
        cadence_days=cadence,
        typical_amount=st.median(amounts),
        last_amount=amounts[-1],
        last_seen=last_seen.isoformat(),
        active=last_seen >= ledger_end - timedelta(days=cadence * ACTIVE_CADENCE_SLACK),
        occurrences=len(rows),
    )


def detect_series(conn: sqlite3.Connection) -> int:
    """Rebuild recurring_series from transactions. Returns rows written.

    Idempotent: the table is rebuilt wholesale, and transactions.recurring_id is
    re-pointed, so running this after every ingest is safe.
    """
    end_row = conn.execute("SELECT MAX(posted_date) FROM transactions").fetchone()
    if not end_row or not end_row[0]:
        return 0
    ledger_end = date.fromisoformat(end_row[0])

    placeholders = ",".join("?" * len(SERIES_TYPES))
    merchants = [
        r[0]
        for r in conn.execute(
            f"""SELECT merchant_id FROM transactions
                WHERE merchant_id IS NOT NULL AND type IN ({placeholders})
                GROUP BY merchant_id HAVING COUNT(*) >= ?""",
            (*SERIES_TYPES, MIN_OCCURRENCES),
        )
    ]

    conn.execute("UPDATE transactions SET recurring_id = NULL")
    conn.execute("DELETE FROM recurring_series")

    written = 0
    for merchant_id in merchants:
        rows = conn.execute(
            f"""SELECT posted_date, amount FROM transactions
                WHERE merchant_id = ? AND type IN ({placeholders})
                ORDER BY posted_date""",
            (merchant_id, *SERIES_TYPES),
        ).fetchall()

        series = _analyze([(r[0], r[1]) for r in rows], ledger_end)
        if series is None:
            continue

        series_id = conn.execute(
            """INSERT INTO recurring_series
               (merchant_id, cadence_days, typical_amount, last_amount, last_seen, active)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (merchant_id, series.cadence_days, series.typical_amount,
             series.last_amount, series.last_seen, int(series.active)),
        ).lastrowid

        conn.execute(
            f"""UPDATE transactions SET recurring_id = ?
                WHERE merchant_id = ? AND type IN ({placeholders})""",
            (series_id, merchant_id, *SERIES_TYPES),
        )
        written += 1

    conn.commit()
    return written


def price_hikes(conn: sqlite3.Connection) -> list[dict]:
    """Active series whose latest charge exceeds typical by more than 5% — §7."""
    return [
        {
            "merchant": row[0],
            "typical_amount": row[1],
            "last_amount": row[2],
            "ratio": abs(row[2] / row[1]) if row[1] else None,
            "last_seen": row[3],
            "cadence_days": row[4],
        }
        for row in conn.execute(
            """SELECT m.canonical_name, s.typical_amount, s.last_amount,
                      s.last_seen, s.cadence_days
               FROM recurring_series s JOIN merchants m ON m.id = s.merchant_id
               WHERE s.active = 1
                 AND ABS(s.last_amount) > ABS(s.typical_amount) * ?
               ORDER BY ABS(s.last_amount / s.typical_amount) DESC""",
            (PRICE_HIKE_RATIO,),
        )
    ]
