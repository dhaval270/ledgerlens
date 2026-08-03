"""Anomaly detection — §6.6. Pure statistics, no network."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from ledgerlens.agent.graph import route_after_planner
from ledgerlens.agent.nodes.anomaly_tool import (
    MIN_ABSOLUTE_DEVIATION,
    MIN_RELATIVE_DEVIATION,
    Z_THRESHOLD,
    _is_material,
    detect,
)
from ledgerlens.db import SCHEMA_PATH
from ledgerlens.seed import seed

START = date(2025, 8, 1)


@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "anom.db")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA_PATH.read_text())
    seed(c)
    c.execute("INSERT INTO accounts (id, name) VALUES (1, 'Test')")
    yield c
    c.close()


def month(conn, category: str, index: int, total: float, n: int = 4):
    """Write `n` purchases summing to `total` in the index-th month."""
    cid = conn.execute("SELECT id FROM categories WHERE name = ?", (category,)).fetchone()[0]
    y, m = divmod(START.month - 1 + index, 12)
    when = date(START.year + y, m + 1, 5).isoformat()
    for i in range(n):
        conn.execute(
            """INSERT INTO transactions
               (account_id, posted_date, amount, raw_descriptor, category_id, type,
                source_file, content_hash, created_at)
               VALUES (1, ?, ?, ?, ?, 'purchase', 'test', ?, '2026-01-01')""",
            (when, -total / n, f"{category} {index}-{i}", cid,
             f"{category}-{index}-{i}"),
        )
    conn.commit()


# --- detection ---------------------------------------------------------------

def test_stable_spend_produces_no_findings(conn):
    for i in range(8):
        month(conn, "groceries", i, 400.0)
    assert detect(conn) == []


def test_large_spike_is_detected(conn):
    for i in range(6):
        month(conn, "travel", i, 200.0)
    month(conn, "travel", 6, 3000.0)

    findings = detect(conn)
    assert findings
    top = findings[0]
    assert top.category == "travel"
    assert top.direction == "above"
    # z is undefined against a perfectly flat baseline; severity is not.
    assert top.severity > Z_THRESHOLD


def test_month_is_not_part_of_its_own_baseline(conn):
    """Including it drags the mean toward the outlier and hides the spike."""
    for i in range(6):
        month(conn, "travel", i, 200.0)
    month(conn, "travel", 6, 3000.0)

    top = detect(conn)[0]
    assert top.baseline_mean == pytest.approx(200.0, abs=0.01)
    assert top.baseline_months == 6


def test_insufficient_history_is_skipped(conn):
    """Two months cannot establish a baseline."""
    month(conn, "travel", 0, 200.0)
    month(conn, "travel", 1, 3000.0)
    assert detect(conn) == []


def test_drop_is_flagged_as_below(conn):
    for i in range(6):
        month(conn, "groceries", i, 800.0)
    month(conn, "groceries", 6, 100.0)

    top = detect(conn)[0]
    assert top.direction == "below"


def test_category_filter(conn):
    for i in range(6):
        month(conn, "travel", i, 200.0)
        month(conn, "dining", i, 100.0)
    month(conn, "travel", 6, 3000.0)
    month(conn, "dining", 6, 900.0)

    assert {f.category for f in detect(conn, category="travel")} == {"travel"}


# --- materiality -------------------------------------------------------------

def test_tiny_absolute_change_is_not_material():
    """A near-constant category collapses its IQR fences; $1.50 is not news."""
    assert not _is_material(52.27, 50.77)


def test_tiny_relative_change_is_not_material():
    assert not _is_material(10_030.0, 10_000.0)


def test_large_change_is_material():
    assert _is_material(3000.0, 200.0)


def test_materiality_gates_are_both_required():
    below_absolute = MIN_ABSOLUTE_DEVIATION - 1
    assert not _is_material(10 + below_absolute, 10.0)   # 240% but only $24
    assert not _is_material(10_000 * (1 + MIN_RELATIVE_DEVIATION / 2), 10_000.0)


def test_near_constant_series_produces_no_findings(conn):
    """The subscriptions case: statistically extreme, practically irrelevant."""
    for i in range(6):
        month(conn, "subscriptions", i, 50.47, n=1)
    month(conn, "subscriptions", 6, 52.27, n=1)
    assert detect(conn) == []


# --- output contract ---------------------------------------------------------

def test_findings_are_structured_not_prose(conn):
    for i in range(6):
        month(conn, "travel", i, 200.0)
    month(conn, "travel", 6, 3000.0)

    row = detect(conn)[0].as_dict()
    assert set(row) == {
        "category", "period", "observed", "baseline_mean",
        "baseline_months", "z_score", "direction", "method",
    }


def test_findings_are_ranked_by_severity(conn):
    for i in range(6):
        month(conn, "travel", i, 200.0)
        month(conn, "dining", i, 100.0)
    month(conn, "travel", 6, 3000.0)
    month(conn, "dining", 6, 400.0)

    findings = detect(conn)
    assert findings[0].severity >= findings[-1].severity


# --- routing -----------------------------------------------------------------

def test_anomaly_plan_routes_to_the_anomaly_tool():
    state = {"answerable": True, "plan": [{"tool": "anomaly"}]}
    assert route_after_planner(state) == "anomaly_tool"
