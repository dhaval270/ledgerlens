"""Human-in-the-loop approvals — §8. No LLM anywhere in this path."""

from __future__ import annotations

import sqlite3

import pytest

from ledgerlens.approvals import ApprovalError, StaleProposal, decide, pending, propose
from ledgerlens.db import SCHEMA_PATH
from ledgerlens.seed import seed


@pytest.fixture
def db(tmp_path):
    """A small ledger on disk. Returns the path; open it to assert on writes."""
    path = tmp_path / "approvals.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_PATH.read_text())
    seed(conn)
    conn.execute("INSERT INTO accounts (id, name) VALUES (1, 'Test')")
    conn.commit()
    conn.close()
    return path


def open_db(path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def category_id(conn, name: str) -> int:
    return conn.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()[0]


def add_merchant(path, name: str, category: str | None = None) -> int:
    conn = open_db(path)
    cid = category_id(conn, category) if category else None
    mid = conn.execute(
        "INSERT INTO merchants (canonical_name, default_category_id) VALUES (?, ?)",
        (name, cid),
    ).lastrowid
    conn.commit()
    conn.close()
    return mid


def add_txn(path, merchant_id: int, when: str, amount: float, category: str, tag: str = "") -> None:
    conn = open_db(path)
    conn.execute(
        """INSERT INTO transactions
           (account_id, posted_date, amount, raw_descriptor, merchant_id, category_id,
            type, source_file, content_hash, created_at)
           VALUES (1, ?, ?, ?, ?, ?, 'purchase', 'test', ?, '2026-01-01')""",
        (when, amount, f"d{tag}", merchant_id, category_id(conn, category),
         f"{merchant_id}-{when}-{amount}-{tag}"),
    )
    conn.commit()
    conn.close()


def count(path, sql: str, *args) -> int:
    conn = open_db(path)
    try:
        return conn.execute(sql, args).fetchone()[0]
    finally:
        conn.close()


# --- the property the whole section exists for -------------------------------

def test_proposing_writes_nothing(db):
    """The diff is shown before anything happens, not after."""
    m = add_merchant(db, "Peloton", "shopping")
    add_txn(db, m, "2026-07-05", -44.0, "shopping")

    propose("recategorize", db_path=str(db), merchant_id=m, category="health")

    assert count(db, "SELECT COUNT(*) FROM corrections") == 0
    assert count(db, "SELECT default_category_id FROM merchants WHERE id = ?", m) == \
        count(db, "SELECT id FROM categories WHERE name = 'shopping'")


def test_rejecting_writes_nothing(db):
    m = add_merchant(db, "Peloton", "shopping")
    add_txn(db, m, "2026-07-05", -44.0, "shopping")

    handle = propose("recategorize", db_path=str(db), merchant_id=m, category="health")
    outcome = decide(handle["thread_id"], approved=False)

    assert outcome["status"] == "rejected"
    assert outcome["result"] is None
    assert count(db, "SELECT COUNT(*) FROM corrections") == 0


def test_approving_applies_the_change(db):
    m = add_merchant(db, "Peloton", "shopping")
    for i in range(3):
        add_txn(db, m, f"2026-07-0{i+1}", -44.0, "shopping", tag=str(i))

    handle = propose("recategorize", db_path=str(db), merchant_id=m, category="health")
    outcome = decide(handle["thread_id"], approved=True)

    assert outcome["status"] == "applied"
    assert outcome["result"]["transactions_updated"] == 3
    assert count(db, "SELECT COUNT(*) FROM corrections") == 1
    assert count(
        db,
        """SELECT COUNT(*) FROM transactions t JOIN categories c ON c.id = t.category_id
           WHERE c.name = 'health'""",
    ) == 3


def test_thread_is_paused_until_decided(db):
    m = add_merchant(db, "Peloton", "shopping")
    add_txn(db, m, "2026-07-05", -44.0, "shopping")

    handle = propose("recategorize", db_path=str(db), merchant_id=m, category="health")
    assert pending(handle["thread_id"])["action"] == "recategorize"

    decide(handle["thread_id"], approved=True)
    assert pending(handle["thread_id"]) is None


# --- the diff must still be the diff -----------------------------------------

def test_a_changed_ledger_invalidates_the_proposal(db):
    """Approval is for one specific diff, not standing permission."""
    m = add_merchant(db, "Peloton", "shopping")
    add_txn(db, m, "2026-07-05", -44.0, "shopping", tag="a")

    handle = propose("recategorize", db_path=str(db), merchant_id=m, category="health")
    add_txn(db, m, "2026-07-12", -44.0, "shopping", tag="b")   # an ingest lands

    with pytest.raises(StaleProposal):
        decide(handle["thread_id"], approved=True)

    assert count(db, "SELECT COUNT(*) FROM corrections") == 0


def test_reproposing_after_a_change_succeeds(db):
    m = add_merchant(db, "Peloton", "shopping")
    add_txn(db, m, "2026-07-05", -44.0, "shopping", tag="a")
    propose("recategorize", db_path=str(db), merchant_id=m, category="health")
    add_txn(db, m, "2026-07-12", -44.0, "shopping", tag="b")

    fresh = propose("recategorize", db_path=str(db), merchant_id=m, category="health")
    assert fresh["proposal"]["affected_rows"] == 2
    assert decide(fresh["thread_id"], approved=True)["status"] == "applied"


# --- proposals that cannot be applied fail in front of the user --------------

def test_a_no_op_recategorize_is_refused_at_propose_time(db):
    m = add_merchant(db, "Peloton", "health")
    with pytest.raises(ApprovalError, match="already"):
        propose("recategorize", db_path=str(db), merchant_id=m, category="health")


def test_unknown_category_is_refused(db):
    m = add_merchant(db, "Peloton", "shopping")
    with pytest.raises(ApprovalError, match="unknown category"):
        propose("recategorize", db_path=str(db), merchant_id=m, category="crypto")


def test_unknown_action_is_refused(db):
    with pytest.raises(ApprovalError, match="unknown action"):
        propose("delete_everything", db_path=str(db))


# --- merge merchants ---------------------------------------------------------

def test_merge_moves_transactions_and_removes_the_source(db):
    keep = add_merchant(db, "Amazon", "shopping")
    dupe = add_merchant(db, "Amazon Mktp", "shopping")
    add_txn(db, keep, "2026-07-01", -20.0, "shopping", tag="k")
    add_txn(db, dupe, "2026-07-02", -30.0, "shopping", tag="d")

    handle = propose("merge_merchants", db_path=str(db), source_id=dupe, target_id=keep)
    assert handle["proposal"]["after"]["transactions"] == 2

    decide(handle["thread_id"], approved=True)

    assert count(db, "SELECT COUNT(*) FROM transactions WHERE merchant_id = ?", keep) == 2
    assert count(db, "SELECT COUNT(*) FROM merchants WHERE id = ?", dupe) == 0


def test_merge_into_self_is_refused(db):
    m = add_merchant(db, "Amazon", "shopping")
    with pytest.raises(ApprovalError, match="itself"):
        propose("merge_merchants", db_path=str(db), source_id=m, target_id=m)


def test_merge_does_not_overwrite_an_explicit_target_category(db):
    keep = add_merchant(db, "Amazon", "shopping")
    dupe = add_merchant(db, "Amazon Mktp", "groceries")
    handle = propose("merge_merchants", db_path=str(db), source_id=dupe, target_id=keep)
    decide(handle["thread_id"], approved=True)

    conn = open_db(db)
    try:
        assert conn.execute(
            "SELECT default_category_id FROM merchants WHERE id = ?", (keep,)
        ).fetchone()[0] == category_id(conn, "shopping")
    finally:
        conn.close()


# --- budgets -----------------------------------------------------------------

def test_creating_a_budget_requires_approval(db):
    handle = propose("set_budget", db_path=str(db), category="dining", limit_amount=300.0)
    assert handle["proposal"]["before"]["limit"] is None
    assert count(db, "SELECT COUNT(*) FROM budgets") == 0

    decide(handle["thread_id"], approved=True)
    assert count(db, "SELECT COUNT(*) FROM budgets") == 1


def test_adjusting_a_budget_shows_the_old_limit(db):
    first = propose("set_budget", db_path=str(db), category="dining", limit_amount=300.0)
    decide(first["thread_id"], approved=True)

    second = propose("set_budget", db_path=str(db), category="dining", limit_amount=250.0)
    assert second["proposal"]["before"]["limit"] == 300.0

    decide(second["thread_id"], approved=True)
    assert count(db, "SELECT COUNT(*) FROM budgets") == 1
    assert count(db, "SELECT limit_amount FROM budgets") == 250.0


def test_a_negative_budget_is_refused(db):
    with pytest.raises(ApprovalError, match="positive"):
        propose("set_budget", db_path=str(db), category="dining", limit_amount=-50.0)


# --- insights ----------------------------------------------------------------

def _insight(path) -> int:
    conn = open_db(path)
    iid = conn.execute(
        """INSERT INTO insights (kind, subject_id, period, payload, surfaced_at)
           VALUES ('price_hike', 1, '2026-07', '{"merchant": "Spotify"}', '2026-08-01')"""
    ).lastrowid
    conn.commit()
    conn.close()
    return iid


def test_dismissing_an_insight_requires_approval(db):
    iid = _insight(db)
    handle = propose("dismiss_insight", db_path=str(db), insight_id=iid)
    assert "Spotify" in handle["proposal"]["summary"]
    assert count(db, "SELECT dismissed FROM insights WHERE id = ?", iid) == 0

    decide(handle["thread_id"], approved=True)
    assert count(db, "SELECT dismissed FROM insights WHERE id = ?", iid) == 1


def test_dismissing_twice_is_refused(db):
    iid = _insight(db)
    decide(propose("dismiss_insight", db_path=str(db), insight_id=iid)["thread_id"], True)

    with pytest.raises(ApprovalError, match="already dismissed"):
        propose("dismiss_insight", db_path=str(db), insight_id=iid)
