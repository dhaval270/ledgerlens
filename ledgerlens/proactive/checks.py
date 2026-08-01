"""Deterministic detectors — §7. No LLM in this module.

  price_hike           recurring_series.last_amount > typical_amount * 1.05
  duplicate_charge     same merchant + same amount within 48h
  category_overspend   month-to-date sum > budgets.limit_amount
  baseline_anomaly     category z-score > 2 vs trailing 6 months
  new_merchant         first transactions row for a merchant
  dormant_subscription series active but category unused for 90 days

Each writes to insights, where UNIQUE(kind, subject_id, period) prevents repeats.
"""

from __future__ import annotations

import sqlite3

PRICE_HIKE_RATIO = 1.05
DUPLICATE_WINDOW_HOURS = 48
Z_THRESHOLD = 2.0
BASELINE_MONTHS = 6
DORMANT_DAYS = 90


def run_all(conn: sqlite3.Connection, period: str) -> list[dict]:
    """Run every detector for a YYYY-MM period. Returns insights written."""
    raise NotImplementedError
