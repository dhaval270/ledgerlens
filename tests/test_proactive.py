"""Proactive detectors — §7. Deterministic; no LLM in any of these."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from ledgerlens.db import SCHEMA_PATH
from ledgerlens.proactive.checks import (
    check_category_overspend,
    check_dormant_subscriptions,
    check_duplicate_charges,
    check_new_merchants,
    check_price_hikes,
    run_all,
    undismissed,
)
from ledgerlens.proactive.digest import digest
from ledgerlens.seed import seed

START = date(2025, 8, 1)


@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "proactive.db")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA_PATH.read_text())
    seed(c)
    c.execute("INSERT INTO accounts (id, name) VALUES (1, 'Test')")
    yield c
    c.close()


def merchant(conn, name: str, category: str | None = None) -> int:
    cid = None
    if category:
        cid = conn.execute("SELECT id FROM categories WHERE name=?", (category,)).fetchone()[0]
    return conn.execute(
        "INSERT INTO merchants (canonical_name, default_category_id) VALUES (?, ?)",
        (name, cid),
    ).lastrowid


def txn(conn, merchant_id, when: str, amount: float, category: str | None = None,
        recurring_id=None, tag: str = ""):
    cid = None
    if category:
        cid = conn.execute("SELECT id FROM categories WHERE name=?", (category,)).fetchone()[0]
    conn.execute(
        """INSERT INTO transactions
           (account_id, posted_date, amount, raw_descriptor, merchant_id, category_id,
            type, recurring_id, source_file, content_hash, created_at)
           VALUES (1,?,?,?,?,?, 'purchase', ?, 'test', ?, '2026-01-01')""",
        (when, amount, f"d{tag}{when}{amount}", merchant_id, cid, recurring_id,
         f"{merchant_id}-{when}-{amount}-{tag}"),
    )
    conn.commit()


# --- idempotency: the property that makes re-running safe -------------------

def test_reruns_do_not_duplicate_insights(conn):
    m = merchant(conn, "Amazon", "shopping")
    txn(conn, m, "2026-03-05", -31.21, "shopping", tag="a")
    txn(conn, m, "2026-03-06", -31.21, "shopping", tag="b")

    first = check_duplicate_charges(conn, "2026-03")
    second = check_duplicate_charges(conn, "2026-03")
    conn.commit()

    assert first == 1 and second == 0
    (n,) = conn.execute("SELECT COUNT(*) FROM insights WHERE kind='duplicate_charge'").fetchone()
    assert n == 1


# --- duplicate charges -------------------------------------------------------

def test_same_amount_within_48h_is_flagged(conn):
    m = merchant(conn, "Amazon", "shopping")
    txn(conn, m, "2026-03-05", -31.21, "shopping", tag="a")
    txn(conn, m, "2026-03-06", -31.21, "shopping", tag="b")
    assert check_duplicate_charges(conn, "2026-03") == 1


def test_same_amount_outside_48h_is_not_flagged(conn):
    m = merchant(conn, "Amazon", "shopping")
    txn(conn, m, "2026-03-01", -31.21, "shopping", tag="a")
    txn(conn, m, "2026-03-09", -31.21, "shopping", tag="b")
    assert check_duplicate_charges(conn, "2026-03") == 0


def test_duplicate_belongs_only_to_its_own_period(conn):
    """A charge in March must not resurface in every later digest."""
    m = merchant(conn, "Amazon", "shopping")
    txn(conn, m, "2026-03-05", -31.21, "shopping", tag="a")
    txn(conn, m, "2026-03-06", -31.21, "shopping", tag="b")
    assert check_duplicate_charges(conn, "2026-05") == 0


# --- new merchants -----------------------------------------------------------

def test_new_merchant_in_a_later_period_is_flagged(conn):
    old = merchant(conn, "Trader Joe's", "groceries")
    txn(conn, old, "2026-01-05", -40.0, "groceries")
    new = merchant(conn, "Peloton", "health")
    txn(conn, new, "2026-03-12", -44.0, "health")

    assert check_new_merchants(conn, "2026-03") == 1


def test_first_period_flags_nothing(conn):
    """Everything is new in month one; saying so is noise, not insight."""
    a = merchant(conn, "Amazon", "shopping")
    b = merchant(conn, "Target", "shopping")
    txn(conn, a, "2026-01-05", -40.0, "shopping")
    txn(conn, b, "2026-01-06", -20.0, "shopping")

    assert check_new_merchants(conn, "2026-01") == 0


# --- dormant subscriptions ---------------------------------------------------

def _series(conn, merchant_id, amount=-24.99, last_seen="2026-06-05"):
    return conn.execute(
        """INSERT INTO recurring_series
           (merchant_id, cadence_days, typical_amount, last_amount, last_seen, active)
           VALUES (?, 30, ?, ?, ?, 1)""",
        (merchant_id, amount, amount, last_seen),
    ).lastrowid


def test_category_that_was_never_used_is_not_dormant(conn):
    """Rent contains nothing but rent; its silence is normal, not a signal."""
    m = merchant(conn, "Greenline", "rent")
    _series(conn, m, -1450.0)
    txn(conn, m, "2026-06-01", -1450.0, "rent")
    conn.commit()
    assert check_dormant_subscriptions(conn, "2026-06") == 0


def test_previously_active_category_gone_quiet_is_dormant(conn):
    gym = merchant(conn, "Planet Fitness", "health")
    sid = _series(conn, gym)
    txn(conn, gym, "2026-06-05", -24.99, "health", recurring_id=sid)

    other = merchant(conn, "CVS", "health")
    txn(conn, other, "2025-09-01", -30.0, "health")   # historic, well before cutoff
    conn.commit()

    assert check_dormant_subscriptions(conn, "2026-06") == 1


def test_recently_used_category_is_not_dormant(conn):
    gym = merchant(conn, "Planet Fitness", "health")
    sid = _series(conn, gym)
    txn(conn, gym, "2026-06-05", -24.99, "health", recurring_id=sid)

    other = merchant(conn, "CVS", "health")
    txn(conn, other, "2025-09-01", -30.0, "health")
    txn(conn, other, "2026-05-20", -18.0, "health", tag="recent")
    conn.commit()

    assert check_dormant_subscriptions(conn, "2026-06") == 0


# --- budgets -----------------------------------------------------------------

def test_no_budget_means_no_overspend_finding(conn):
    m = merchant(conn, "Amazon", "shopping")
    txn(conn, m, "2026-03-05", -5000.0, "shopping")
    assert check_category_overspend(conn, "2026-03") == 0


def test_exceeding_a_budget_is_flagged(conn):
    cid = conn.execute("SELECT id FROM categories WHERE name='shopping'").fetchone()[0]
    conn.execute(
        "INSERT INTO budgets (category_id, limit_amount, active_from) VALUES (?, 100.0, '2025-01-01')",
        (cid,),
    )
    m = merchant(conn, "Amazon", "shopping")
    txn(conn, m, "2026-03-05", -250.0, "shopping")
    conn.commit()

    assert check_category_overspend(conn, "2026-03") == 1


def test_staying_under_budget_is_not_flagged(conn):
    cid = conn.execute("SELECT id FROM categories WHERE name='shopping'").fetchone()[0]
    conn.execute(
        "INSERT INTO budgets (category_id, limit_amount, active_from) VALUES (?, 500.0, '2025-01-01')",
        (cid,),
    )
    m = merchant(conn, "Amazon", "shopping")
    txn(conn, m, "2026-03-05", -250.0, "shopping")
    conn.commit()

    assert check_category_overspend(conn, "2026-03") == 0


# --- price hikes -------------------------------------------------------------

def test_price_hike_is_attributed_to_the_period_it_occurred(conn):
    m = merchant(conn, "Spotify", "subscriptions")
    _series(conn, m, amount=-11.99, last_seen="2026-06-08")
    conn.execute("UPDATE recurring_series SET last_amount = -13.79")
    conn.commit()

    assert check_price_hikes(conn, "2026-06") == 1
    assert check_price_hikes(conn, "2026-07") == 0


# --- digest ------------------------------------------------------------------

def test_digest_with_no_findings_needs_no_llm(conn):
    d = digest(conn, "2026-03")
    assert d["count"] == 0
    assert "Nothing notable" in d["summary"]


def test_run_all_reports_every_kind(conn):
    counts = run_all(conn, "2026-03")
    assert set(counts) == {
        "price_hike", "duplicate_charge", "category_overspend",
        "baseline_anomaly", "new_merchant", "dormant_subscription",
    }


def test_dismissed_insights_are_excluded(conn):
    m = merchant(conn, "Amazon", "shopping")
    txn(conn, m, "2026-03-05", -31.21, "shopping", tag="a")
    txn(conn, m, "2026-03-06", -31.21, "shopping", tag="b")
    check_duplicate_charges(conn, "2026-03")
    conn.commit()

    assert len(undismissed(conn, "2026-03")) == 1
    conn.execute("UPDATE insights SET dismissed = 1")
    conn.commit()
    assert undismissed(conn, "2026-03") == []
