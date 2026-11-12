"""Categorization tier tests — §5.4.

All of these exercise tiers 1-3, which must never call an LLM. If any of these
start needing a network connection, the tier ordering has regressed.
"""

from __future__ import annotations

import sqlite3

import pytest

from ledgerlens.db import SCHEMA_PATH
from ledgerlens.ingest.categorize import (
    CONFIDENCE_THRESHOLD,
    categorize,
    record_correction,
)
from ledgerlens.seed import seed


@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "cat.db")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA_PATH.read_text())
    c.execute("PRAGMA foreign_keys = ON")
    seed(c)
    yield c
    c.close()


@pytest.fixture
def merchant(conn):
    return conn.execute(
        "INSERT INTO merchants (canonical_name) VALUES ('Blue Bottle Coffee')"
    ).lastrowid


def _cid(conn, name: str) -> int:
    return conn.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()[0]


# --- tier 3: rules and types -------------------------------------------------

def test_type_implies_category_without_inference(conn):
    for txn_type, expected in (("income", "income"), ("transfer", "transfer"), ("fee", "fees")):
        r = categorize(conn, "ANYTHING AT ALL", None, txn_type=txn_type, use_llm=False)
        assert r.category_name == expected
        assert r.categorized_by == "rule"


def test_regex_rule_matches(conn, merchant):
    r = categorize(conn, "SQ *BLUE BOTTLE COFFEE AUSTIN TX", merchant, use_llm=False)
    assert r.category_name == "coffee"
    assert r.tier == 3


def test_pharmacy_is_health_not_shopping(conn):
    """Rule priority matters: CVS matches both a health and a shopping intent."""
    r = categorize(conn, "CVS/PHARMACY #56138 BOSTON", None, use_llm=False)
    assert r.category_name == "health"


def test_unmatched_falls_back_without_llm(conn):
    r = categorize(conn, "ZZQQ UNRECOGNIZABLE 99", None, use_llm=False)
    assert r.category_name == "uncategorized"


# --- tier 2: learned ---------------------------------------------------------

def test_rule_hit_promotes_to_merchant_default(conn, merchant):
    first = categorize(conn, "SQ *BLUE BOTTLE COFFEE AUSTIN TX", merchant, use_llm=False)
    assert first.tier == 3

    # A descriptor no rule matches now resolves via the learned default.
    second = categorize(conn, "SQ *BLUE BOTTLE XYZ", merchant, use_llm=False)
    assert second.tier == 2
    assert second.categorized_by == "learned"
    assert second.category_name == first.category_name


# --- tier 1: corrections -----------------------------------------------------

def test_correction_outranks_learned(conn, merchant):
    categorize(conn, "SQ *BLUE BOTTLE COFFEE AUSTIN TX", merchant, use_llm=False)  # learns 'coffee'

    record_correction(conn, category_id=_cid(conn, "dining"), merchant_id=merchant)

    r = categorize(conn, "SQ *BLUE BOTTLE COFFEE AUSTIN TX", merchant, use_llm=False)
    assert r.category_name == "dining"
    assert r.categorized_by == "user"
    assert r.tier == 1


def test_correction_clears_stale_learned_default(conn, merchant):
    """A correction that left default_category_id stale would be silently ignored."""
    categorize(conn, "SQ *BLUE BOTTLE COFFEE AUSTIN TX", merchant, use_llm=False)
    record_correction(conn, category_id=_cid(conn, "dining"), merchant_id=merchant)

    default_id = conn.execute(
        "SELECT default_category_id FROM merchants WHERE id = ?", (merchant,)
    ).fetchone()[0]
    assert default_id == _cid(conn, "dining")


# --- review queue ------------------------------------------------------------

def test_low_confidence_is_flagged_for_review(conn):
    r = categorize(conn, "ZZQQ UNRECOGNIZABLE 99", None, use_llm=False)
    assert r.confidence is not None and r.confidence < CONFIDENCE_THRESHOLD
    assert r.needs_review


def test_confident_results_are_not_flagged(conn, merchant):
    r = categorize(conn, "SQ *BLUE BOTTLE COFFEE AUSTIN TX", merchant, use_llm=False)
    assert not r.needs_review
