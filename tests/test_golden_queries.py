"""Golden query set integrity — §9.1.

The set is only useful if its expected values are true. These tests catch the
two ways it rots: malformed entries, and expected values that no longer match
what the reference SQL returns.
"""

from __future__ import annotations

import json

import pytest

from ledgerlens.db import DB_PATH, connect_readonly

GOLDEN = __import__("pathlib").Path(__file__).parent.parent / "evals" / "golden_queries.jsonl"

REQUIRED_BUCKETS = {
    "agg": "simple aggregates",
    "date": "date ranges",
    "cmp": "comparisons",
    "mer": "merchant-specific",
    "rec": "recurring/anomaly",
    "multi": "multi-hop",
    "sem": "semantic recall",
    "refuse": "unanswerable",
}


@pytest.fixture(scope="module")
def entries():
    if not GOLDEN.exists():
        pytest.skip("golden_queries.jsonl not built")
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def test_set_is_large_enough(entries):
    """§9.1 asks for roughly 50."""
    assert len(entries) >= 50


def test_every_bucket_is_represented(entries):
    seen = {e["id"].split("-")[0] for e in entries}
    missing = set(REQUIRED_BUCKETS) - seen
    assert not missing, f"missing coverage: {[REQUIRED_BUCKETS[m] for m in missing]}"


def test_ids_are_unique(entries):
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))


def test_entries_are_well_formed(entries):
    for e in entries:
        assert e["question"].strip()
        assert isinstance(e["tools"], list)
        if e["expect_refusal"]:
            assert e["expected_value"] is None
            assert e["tools"] == [], f"{e['id']}: a refusal should route to no tool"
        else:
            assert e["expected_value"] is not None
            assert e["reference_sql"].strip()


def test_refusals_are_present(entries):
    """§9.1 explicitly wants questions whose correct answer is a refusal."""
    assert sum(e["expect_refusal"] for e in entries) >= 5


@pytest.mark.skipif(not DB_PATH.exists(), reason="ledger.db not built")
def test_expected_values_still_match_reference_sql(entries):
    """Catches silent drift between the dataset and the answer key."""
    drifted = []
    with connect_readonly() as conn:
        for e in entries:
            if e["expect_refusal"]:
                continue
            got = conn.execute(e["reference_sql"]).fetchone()[0]
            if got != e["expected_value"]:
                drifted.append((e["id"], e["expected_value"], got))
    assert not drifted, f"expected values drifted: {drifted}"
