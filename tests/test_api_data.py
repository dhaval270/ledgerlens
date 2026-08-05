"""Read-only data endpoints — /stats and /transactions.

Exercised against the ledger on disk, since both are read-only and the suite
already depends on it existing for the golden-query drift test.
"""

from __future__ import annotations

import pytest

from ledgerlens.db import DB_PATH

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="ledger.db not built")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from ledgerlens.api.main import app

    return TestClient(app)


# --- stats -------------------------------------------------------------------

def test_stats_names_the_open_database(client):
    """Ingesting into one ledger and reading from another is this app's most
    confusing failure, and it produces no error. The path is the answer."""
    body = client.get("/stats").json()
    assert body["db"].endswith(".db")


def test_stats_counts_agree_with_the_type_breakdown(client):
    body = client.get("/stats").json()
    assert sum(body["by_type"].values()) == body["transactions"]


def test_category_totals_cover_purchases_only(client):
    """§4: transfers and income must not appear in a spend breakdown."""
    body = client.get("/stats").json()
    from_categories = sum(c["n"] for c in body["by_category"])
    assert from_categories == body["by_type"].get("purchase", 0)


# --- listing -----------------------------------------------------------------

def test_pages_do_not_overlap_or_skip(client):
    first = client.get("/transactions?limit=20&offset=0").json()
    second = client.get("/transactions?limit=20&offset=20").json()
    ids = [r["id"] for r in first["rows"]] + [r["id"] for r in second["rows"]]
    assert len(set(ids)) == len(ids) == 40


def test_rows_come_back_newest_first(client):
    rows = client.get("/transactions?limit=30").json()["rows"]
    assert rows == sorted(rows, key=lambda r: r["posted_date"], reverse=True)


def test_limit_is_capped(client):
    """An unbounded limit is a way to ask for the whole table by accident."""
    assert client.get("/transactions?limit=99999").json()["limit"] == 500


def test_type_filter_returns_only_that_type(client):
    rows = client.get("/transactions?type=income").json()["rows"]
    assert rows and {r["type"] for r in rows} == {"income"}


def test_search_matches_the_raw_descriptor(client):
    body = client.get("/transactions?q=chipotle").json()
    assert body["total"] > 0
    assert all("chipotle" in r["raw_descriptor"].lower()
               or "chipotle" in (r["merchant"] or "").lower() for r in body["rows"])


def test_sum_describes_the_filtered_set_not_the_page(client):
    """The figure under the table must total every match, not the 50 on screen."""
    body = client.get("/transactions?q=chipotle&limit=5").json()
    assert body["total"] > 5
    assert abs(body["sum"]) > abs(sum(r["amount"] for r in body["rows"]))


def test_a_query_with_sql_syntax_is_treated_as_text(client):
    """Filters are bound parameters. This endpoint takes user text and runs it
    against the same database the agent reads, so it gets the same discipline."""
    body = client.get("/transactions?q=%27%20OR%201%3D1%20--").json()
    assert body["total"] == 0        # matched literally, and matched nothing


def test_an_unknown_type_matches_nothing_rather_than_erroring(client):
    assert client.get("/transactions?type=nonsense").json()["total"] == 0
