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
    "anom": "anomaly routing",
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
            key = e.get("reference_sql") or e.get("reference_fn")
            assert key and key.strip(), f"{e['id']}: no answer key"


def test_refusals_are_present(entries):
    """§9.1 explicitly wants questions whose correct answer is a refusal."""
    assert sum(e["expect_refusal"] for e in entries) >= 5


@pytest.mark.skipif(not DB_PATH.exists(), reason="ledger.db not built")
def test_expected_values_still_match_reference_sql(entries):
    """Catches silent drift between the dataset and the answer key."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evals.build_golden_queries import DETECTORS

    drifted = []
    with connect_readonly() as conn:
        for e in entries:
            if e["expect_refusal"]:
                continue
            # Anomaly keys come from the detector, not from SQL — §6.6's
            # baseline is procedural and has no scalar-SELECT equivalent.
            if fn := e.get("reference_fn"):
                got = DETECTORS[fn](conn)
            else:
                got = conn.execute(e["reference_sql"]).fetchone()[0]
            if got != e["expected_value"]:
                drifted.append((e["id"], e["expected_value"], got))
    assert not drifted, f"expected values drifted: {drifted}"


def test_routing_coverage_is_not_token(entries):
    """The set must exercise more than text-to-SQL.

    Guards a regression that already happened once and went unnoticed for the
    life of the project: `run_golden_eval.py` called the SQL tool directly, so
    routing was never scored, and nobody noticed the set had drifted to 41
    SQL-only queries against 5 that touched anything else. A count here makes
    the imbalance visible in CI rather than in a post-mortem.
    """
    answerable = [e for e in entries if not e["expect_refusal"]]
    semantic = [e for e in answerable if "semantic" in e["tools"]]
    anomaly = [e for e in answerable if "anomaly" in e["tools"]]

    assert len(semantic) >= 8, f"only {len(semantic)} semantic queries"
    assert len(anomaly) >= 6, f"only {len(anomaly)} anomaly queries"


# --- the evals must score the benchmark, not whatever .env points at ---------

def test_every_eval_pins_the_benchmark_ledger():
    """The failure this catches is silent and scores a number anyway.

    `DB_PATH` resolves `LEDGERLENS_DB` at import time, and `.env` sets it for
    whoever is using the app on their own statements. Run from that shell, the
    evals scored the golden set against a personal ledger — sixteen rows, no
    anomalies. `run_verifier_eval.py` crashed with `IndexError`, which was the
    lucky outcome: the golden queries would have returned None and been read as
    regressions. conftest.py has pinned the suite this way from the start; the
    evals never did.
    """
    import pathlib

    evals = pathlib.Path(__file__).resolve().parent.parent / "evals"
    for script in sorted(evals.glob("run_*.py")):
        source = script.read_text()
        pin = source.find("_benchmark")
        first_import = source.find("from ledgerlens")
        if first_import == -1:
            first_import = source.find("import ledgerlens")
        assert pin != -1, f"{script.name} does not pin the benchmark ledger"
        if first_import != -1:
            assert pin < first_import, f"{script.name} pins too late to matter"


def test_the_pin_wins_over_the_environment(monkeypatch):
    """It overrides rather than defaults. "Unset it first" is not a thing
    anyone remembers to do."""
    import importlib
    import os
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    monkeypatch.setenv("LEDGERLENS_DB", "somewhere-else.db")
    module = importlib.reload(importlib.import_module("evals._benchmark"))
    assert os.environ["LEDGERLENS_DB"] == str(module.BENCHMARK_DB)
