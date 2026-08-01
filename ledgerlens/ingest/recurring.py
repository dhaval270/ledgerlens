"""Periodicity detection — §5.5. Pure algorithm, no LLM.

Group by merchant_id, sort dates, compute gaps. A series needs >=3 occurrences
and a gap standard deviation under ~20% of the mean. Store cadence and typical
amount; flag when last_amount deviates from typical_amount by more than 5% —
that is the subscription price-hike detector feeding §7.
"""

from __future__ import annotations

import sqlite3

MIN_OCCURRENCES = 3
MAX_GAP_CV = 0.20        # stddev / mean
PRICE_HIKE_RATIO = 1.05


def detect_series(conn: sqlite3.Connection) -> int:
    """Rebuild recurring_series from transactions. Returns rows written."""
    raise NotImplementedError
