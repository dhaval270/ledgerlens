"""Deterministic detectors — §7. No LLM in this module.

  price_hike           recurring_series.last_amount > typical_amount * 1.05
  duplicate_charge     same merchant + same amount within 48h
  category_overspend   month-to-date sum > budgets.limit_amount
  baseline_anomaly     category z-score > 2 vs trailing 6 months
  new_merchant         first transactions row for a merchant
  dormant_subscription series still charging, category unused for 90 days

Each writes to `insights`, where UNIQUE(kind, subject_id, period) prevents
repeats. Detectors are therefore safe to re-run after every ingest: a finding
already surfaced this period is silently ignored rather than duplicated, which
is what stops a monthly digest from renotifying the same price rise forever.

The statistics live in the tool modules (`agent.nodes.anomaly_tool.detect`,
`ingest.recurring.price_hikes`) and are called from here rather than
reimplemented. A second copy of a z-score is a second thing to get wrong, and
the §6.6 tool and the §7 digest must not be able to disagree about what counts
as an anomaly.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from ..agent.nodes.anomaly_tool import detect as detect_anomalies
from ..ingest.recurring import price_hikes

DUPLICATE_WINDOW_HOURS = 48
DORMANT_DAYS = 90

KINDS = (
    "price_hike",
    "duplicate_charge",
    "category_overspend",
    "baseline_anomaly",
    "new_merchant",
    "dormant_subscription",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(
    conn: sqlite3.Connection, kind: str, subject_id: int | None,
    period: str, payload: dict,
) -> bool:
    """Insert one insight. Returns True if it was new."""
    cur = conn.execute(
        """INSERT OR IGNORE INTO insights (kind, subject_id, period, payload, surfaced_at)
           VALUES (?, ?, ?, ?, ?)""",
        (kind, subject_id, period, json.dumps(payload, default=str), _now()),
    )
    return cur.rowcount > 0


# --- detectors ---------------------------------------------------------------

def check_price_hikes(conn: sqlite3.Connection, period: str) -> int:
    found = 0
    for hike in price_hikes(conn):
        # Attribute the hike to the period it was last charged in, not to every
        # period after it — otherwise one rise renotifies forever.
        if (hike.get("last_seen") or "")[:7] != period:
            continue
        row = conn.execute(
            "SELECT id FROM merchants WHERE canonical_name = ?", (hike["merchant"],)
        ).fetchone()
        found += _record(conn, "price_hike", row[0] if row else None, period, hike)
    return found


def check_duplicate_charges(conn: sqlite3.Connection, period: str) -> int:
    """Same merchant, same amount, within 48h — the classic double charge."""
    rows = conn.execute(
        """SELECT a.id, a.merchant_id, m.canonical_name, a.amount,
                  a.posted_date, b.posted_date
           FROM transactions a
           JOIN transactions b
             ON b.merchant_id = a.merchant_id
            AND b.amount = a.amount
            AND b.id > a.id
            AND julianday(b.posted_date) - julianday(a.posted_date) <= ?
           JOIN merchants m ON m.id = a.merchant_id
           WHERE a.type = 'purchase' AND b.type = 'purchase'
             AND substr(a.posted_date,1,7) = ?
           ORDER BY a.posted_date""",
        (DUPLICATE_WINDOW_HOURS / 24.0, period),
    ).fetchall()

    found = 0
    for txn_id, merchant_id, merchant, amount, first, second in rows:
        found += _record(conn, "duplicate_charge", txn_id, period, {
            "merchant": merchant, "amount": amount, "dates": [first, second],
        })
    return found


def check_category_overspend(conn: sqlite3.Connection, period: str) -> int:
    """Month-to-date spend against an active budget. No budgets, no findings."""
    rows = conn.execute(
        """SELECT b.category_id, c.name, b.limit_amount,
                  ABS(COALESCE(SUM(t.amount), 0)) AS spent
           FROM budgets b
           JOIN categories c ON c.id = b.category_id
           LEFT JOIN transactions t
             ON t.category_id = b.category_id
            AND t.type = 'purchase'
            AND substr(t.posted_date,1,7) = ?
           WHERE b.active_from <= ?
           GROUP BY b.id
           HAVING spent > b.limit_amount""",
        (period, period + "-31"),
    ).fetchall()

    found = 0
    for category_id, name, limit_amount, spent in rows:
        found += _record(conn, "category_overspend", category_id, period, {
            "category": name, "limit": limit_amount, "spent": round(spent, 2),
            "over_by": round(spent - limit_amount, 2),
        })
    return found


def check_baseline_anomalies(conn: sqlite3.Connection, period: str) -> int:
    found = 0
    for finding in detect_anomalies(conn):
        if finding.period != period:
            continue
        row = conn.execute(
            "SELECT id FROM categories WHERE name = ?", (finding.category,)
        ).fetchone()
        found += _record(conn, "baseline_anomaly", row[0] if row else None,
                         period, finding.as_dict())
    return found


def check_new_merchants(conn: sqlite3.Connection, period: str) -> int:
    """A merchant seen for the first time this period.

    The earliest period in the ledger is skipped: every merchant is new in it,
    which produced 17 findings for the first month and told the reader nothing.
    "New" only means something once there is history to be new against.
    """
    (earliest,) = conn.execute(
        "SELECT substr(MIN(posted_date),1,7) FROM transactions"
    ).fetchone()
    if earliest is None or period <= earliest:
        return 0

    rows = conn.execute(
        """SELECT m.id, m.canonical_name, MIN(t.posted_date) AS first_txn
           FROM transactions t JOIN merchants m ON m.id = t.merchant_id
           WHERE t.type = 'purchase'
           GROUP BY m.id
           HAVING substr(first_txn,1,7) = ?""",
        (period,),
    ).fetchall()

    found = 0
    for merchant_id, name, first_seen in rows:
        found += _record(conn, "new_merchant", merchant_id, period, {
            "merchant": name, "first_seen": first_seen,
        })
    return found


def check_dormant_subscriptions(conn: sqlite3.Connection, period: str) -> int:
    """Still charging, but a previously-active category has gone quiet.

    A gym that bills monthly while you have stopped buying anything else
    health-related is the case this is for. Proxy, not proof — the ledger
    records payments, not attendance — so the payload says what was observed.

    The "previously active" requirement is what makes this usable. Judging on
    current silence alone flagged every series in the ledger, rent included:
    categories like rent, utilities and subscriptions contain nothing *but*
    their recurring charge, so absence of other activity is their normal state,
    not a signal. A category only counts as dormant if it used to see other
    spending and now does not.
    """
    cutoff = (datetime.fromisoformat(period + "-01") - timedelta(days=DORMANT_DAYS)).date()

    rows = conn.execute(
        """SELECT s.id, m.canonical_name, c.name, s.typical_amount, s.last_seen
           FROM recurring_series s
           JOIN merchants m ON m.id = s.merchant_id
           LEFT JOIN categories c ON c.id = m.default_category_id
           WHERE s.active = 1 AND s.typical_amount < 0"""
    ).fetchall()

    found = 0
    for series_id, merchant, category, amount, last_seen in rows:
        if not category:
            continue
        recent, historic = conn.execute(
            """SELECT
                 SUM(CASE WHEN t.posted_date >= ? THEN 1 ELSE 0 END),
                 SUM(CASE WHEN t.posted_date <  ? THEN 1 ELSE 0 END)
               FROM transactions t
               JOIN categories c ON c.id = t.category_id
               WHERE c.name = ? AND t.type = 'purchase'
                 AND t.recurring_id IS NULL""",
            (cutoff.isoformat(), cutoff.isoformat(), category),
        ).fetchone()

        if recent:            # still being used
            continue
        if not historic:      # never used beyond the subscription itself
            continue
        found += _record(conn, "dormant_subscription", series_id, period, {
            "merchant": merchant, "category": category,
            "monthly_amount": amount, "last_charged": last_seen,
            "note": f"no other {category} activity in {DORMANT_DAYS} days",
        })
    return found


# --- orchestration -----------------------------------------------------------

def run_all(conn: sqlite3.Connection, period: str) -> dict[str, int]:
    """Run every detector for a YYYY-MM period. Returns new insights per kind."""
    counts = {
        "price_hike": check_price_hikes(conn, period),
        "duplicate_charge": check_duplicate_charges(conn, period),
        "category_overspend": check_category_overspend(conn, period),
        "baseline_anomaly": check_baseline_anomalies(conn, period),
        "new_merchant": check_new_merchants(conn, period),
        "dormant_subscription": check_dormant_subscriptions(conn, period),
    }
    conn.commit()
    return counts


def undismissed(conn: sqlite3.Connection, period: str) -> list[dict]:
    """Insights to narrate — findings only, never raw transactions."""
    return [
        {"id": r[0], "kind": r[1], "period": r[2], **json.loads(r[3])}
        for r in conn.execute(
            """SELECT id, kind, period, payload FROM insights
               WHERE period = ? AND dismissed = 0
               ORDER BY kind, id""",
            (period,),
        )
    ]
