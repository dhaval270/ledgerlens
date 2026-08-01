"""Recurring-series tests — §5.5. Pure algorithm, no network."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from ledgerlens.db import SCHEMA_PATH
from ledgerlens.ingest.recurring import (
    MAX_GAP_CV,
    PRICE_HIKE_RATIO,
    detect_series,
    price_hikes,
)
from ledgerlens.seed import seed

START = date(2025, 8, 1)


@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "rec.db")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA_PATH.read_text())
    c.execute("PRAGMA foreign_keys = ON")
    seed(c)
    c.execute("INSERT INTO accounts (id, name) VALUES (1, 'Test')")
    yield c
    c.close()


def add(conn, merchant: str, offsets: list[int], amounts: list[float], txn_type="purchase"):
    row = conn.execute(
        "SELECT id FROM merchants WHERE canonical_name = ?", (merchant,)
    ).fetchone()
    mid = row[0] if row else conn.execute(
        "INSERT INTO merchants (canonical_name) VALUES (?)", (merchant,)
    ).lastrowid

    for i, (offset, amount) in enumerate(zip(offsets, amounts)):
        when = (START + timedelta(days=offset)).isoformat()
        conn.execute(
            """INSERT INTO transactions
               (account_id, posted_date, amount, raw_descriptor, merchant_id, type,
                source_file, content_hash, created_at)
               VALUES (1, ?, ?, ?, ?, ?, 'test', ?, '2026-01-01')""",
            (when, amount, f"{merchant} {i}", mid, txn_type, f"{merchant}-{i}-{offset}"),
        )
    conn.commit()
    return mid


def test_regular_monthly_series_is_detected(conn):
    add(conn, "Netflix", [0, 30, 60, 90], [-15.49] * 4)
    assert detect_series(conn) == 1

    s = conn.execute("SELECT * FROM recurring_series").fetchone()
    assert s["cadence_days"] == 30
    assert s["typical_amount"] == -15.49
    assert s["active"] == 1


def test_two_occurrences_is_not_a_series(conn):
    """§5.5 requires >=3 — two points define a gap, not a cadence."""
    add(conn, "Netflix", [0, 30], [-15.49] * 2)
    assert detect_series(conn) == 0


def test_irregular_gaps_are_rejected(conn):
    """Everyday spend must not become a subscription."""
    add(conn, "Blue Bottle", [0, 2, 3, 9, 11, 25, 26, 40], [-5.20] * 8)
    assert detect_series(conn) == 0


def test_unstable_amounts_are_rejected(conn):
    """Perfect cadence but wildly varying amounts is not a subscription."""
    add(conn, "Groceries Inc", [0, 30, 60, 90], [-20.0, -95.0, -41.0, -130.0])
    assert detect_series(conn) == 0


def test_same_day_repeats_are_not_a_cadence(conn):
    add(conn, "Duplicate Co", [0, 0, 0, 0], [-10.0] * 4)
    assert detect_series(conn) == 0


def test_price_hike_is_flagged(conn):
    add(conn, "Spotify", [0, 30, 60, 90, 120, 150], [-11.99] * 4 + [-13.79] * 2)
    detect_series(conn)

    hikes = price_hikes(conn)
    assert len(hikes) == 1
    assert hikes[0]["merchant"] == "Spotify"
    assert hikes[0]["ratio"] > PRICE_HIKE_RATIO


def test_stable_price_is_not_flagged(conn):
    add(conn, "Netflix", [0, 30, 60, 90], [-15.49] * 4)
    detect_series(conn)
    assert price_hikes(conn) == []


def test_median_resists_the_hike_it_measures(conn):
    """typical_amount must not drift toward the raised price."""
    add(conn, "Spotify", [0, 30, 60, 90, 120, 150], [-11.99] * 4 + [-13.79] * 2)
    detect_series(conn)
    s = conn.execute("SELECT typical_amount FROM recurring_series").fetchone()
    assert s["typical_amount"] == -11.99  # mean would be -12.59


def test_dormant_series_is_marked_inactive(conn):
    """A series that stopped long ago must not report as active."""
    add(conn, "Old Gym", [0, 30, 60], [-24.99] * 3)
    add(conn, "Current Sub", [0, 30, 60, 90, 120, 150, 180, 210], [-9.99] * 8)
    detect_series(conn)

    rows = {
        r["canonical_name"]: r["active"]
        for r in conn.execute(
            """SELECT m.canonical_name, s.active FROM recurring_series s
               JOIN merchants m ON m.id = s.merchant_id"""
        )
    }
    assert rows["Current Sub"] == 1
    assert rows["Old Gym"] == 0


def test_detect_series_is_idempotent(conn):
    add(conn, "Netflix", [0, 30, 60, 90], [-15.49] * 4)
    assert detect_series(conn) == detect_series(conn) == 1
    (n,) = conn.execute("SELECT COUNT(*) FROM recurring_series").fetchone()
    assert n == 1


def test_transactions_are_linked_to_their_series(conn):
    add(conn, "Netflix", [0, 30, 60, 90], [-15.49] * 4)
    detect_series(conn)
    (linked,) = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE recurring_id IS NOT NULL"
    ).fetchone()
    assert linked == 4


def test_refunds_do_not_corrupt_a_series(conn):
    """A refund against a regular merchant is inflow noise, not an occurrence."""
    mid = add(conn, "Adobe", [0, 30, 60, 90], [-22.99] * 4)
    add(conn, "Adobe", [45], [22.99], txn_type="refund")
    detect_series(conn)

    s = conn.execute("SELECT cadence_days, typical_amount FROM recurring_series").fetchone()
    assert s["cadence_days"] == 30
    assert s["typical_amount"] == -22.99
