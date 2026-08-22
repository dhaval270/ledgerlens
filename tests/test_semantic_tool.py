"""Semantic tool — §6.5.

The load-bearing property is negative: this tool returns transaction IDs and
never a number. If an aggregate ever appears in its output, non-negotiable #1
is broken and the verifier has nothing to reconcile against.
"""

from __future__ import annotations

import numpy as np
import pytest

from ledgerlens.agent.graph import route_after_planner, route_after_semantic
from ledgerlens.agent.nodes import semantic_tool as mod
from ledgerlens.agent.nodes.semantic_tool import INDEX_PATH, search, semantic_tool


@pytest.fixture
def fake_index(tmp_path, monkeypatch):
    """A tiny deterministic index — no model download, no network."""
    ids = np.array([10, 20, 30], dtype=np.int64)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    path = tmp_path / "vec.npz"
    np.savez(path, ids=ids, vectors=vectors)

    class Stub:
        def encode(self, texts, **kw):
            return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)

    monkeypatch.setattr(mod, "_model", lambda: Stub())
    return path


# --- retrieval ---------------------------------------------------------------

def test_search_returns_ids_ranked_by_similarity(fake_index):
    assert search("anything", k=3, index_path=fake_index) == [10, 30, 20]


def test_search_respects_k(fake_index):
    assert len(search("anything", k=2, index_path=fake_index)) == 2


def test_k_larger_than_index_does_not_crash(fake_index):
    assert len(search("anything", k=99, index_path=fake_index)) == 3


def test_missing_index_raises_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ledgerlens.index"):
        search("anything", index_path=tmp_path / "absent.npz")


# --- the non-negotiable ------------------------------------------------------

def test_tool_returns_ids_and_no_rows(fake_index, monkeypatch):
    monkeypatch.setattr(mod, "INDEX_PATH", fake_index)
    monkeypatch.setattr(mod, "search", lambda q, k=20, index_path=None: [10, 20])

    out = semantic_tool({"question": "that trip to Boston", "plan": [], "tool_results": []})
    entry = out["tool_results"][0]

    assert entry["transaction_ids"] == [10, 20]
    assert entry["rows"] == [], "semantic must not supply rows for the verifier"


def test_tool_never_emits_a_numeric_aggregate(fake_index, monkeypatch):
    monkeypatch.setattr(mod, "search", lambda q, k=20, index_path=None: [10, 20, 30])
    out = semantic_tool({"question": "medical things", "plan": [], "tool_results": []})
    entry = out["tool_results"][0]
    assert set(entry) == {"tool", "sub_question", "transaction_ids", "rows", "error"}


def test_search_failure_is_reported_not_raised(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("index corrupt")

    monkeypatch.setattr(mod, "search", boom)
    out = semantic_tool({"question": "q", "plan": [], "tool_results": []})
    entry = out["tool_results"][0]
    assert entry["error"] and entry["transaction_ids"] == []


def test_existing_results_are_preserved(fake_index, monkeypatch):
    monkeypatch.setattr(mod, "search", lambda q, k=20, index_path=None: [1])
    prior = {"tool": "sql", "rows": [{"n": 1}], "error": None}
    out = semantic_tool({"question": "q", "plan": [], "tool_results": [prior]})
    assert len(out["tool_results"]) == 2


# --- routing: semantic runs before sql (a data dependency) -------------------

def test_semantic_step_routes_to_semantic_first():
    state = {"answerable": True, "plan": [{"tool": "semantic"}, {"tool": "sql"}]}
    assert route_after_planner(state) == "semantic_tool"


def test_semantic_hands_off_to_sql_when_a_figure_is_needed():
    state = {"plan": [{"tool": "semantic"}, {"tool": "sql"}]}
    assert route_after_semantic(state) == "sql_tool"


def test_semantic_only_plan_still_reaches_sql():
    """Retrieval alone can never answer, so it must never be the last hop.

    This test previously asserted the opposite — that a semantic-only plan
    routes to `answer` — and passed for the life of the project. It was wrong.
    `semantic_tool` returns an explicitly empty `rows` (§6.5 forbids it
    producing figures), so that path reached the answer node with nothing to
    render and reported "no matching transactions" over a retrieval that had
    actually found the right rows.

    Nothing caught it because the golden harness called `run_query()` directly
    and never built the graph. Scoring end to end surfaced it immediately:
    three of ten semantic queries planned semantic-only, and all three returned
    no_data.
    """
    assert route_after_semantic({"plan": [{"tool": "semantic"}]}) == "sql_tool"


# --- a plan that names two tools must run two tools --------------------------

def _state(plan, tool_results=None):
    return {"question": "q", "plan": plan, "tool_results": tool_results or []}


def test_anomaly_hands_off_to_sql_when_the_plan_asked_for_both():
    """rec-02 planned correctly and ran short.

    "Did any of my subscriptions go up in price?" plans anomaly (which charge
    moved) plus sql (by how much). The graph sent every anomaly result straight
    to the answer node, so the sql step was dropped without a word — the plan
    said two tools and `tool_results` held one.
    """
    from ledgerlens.agent.graph import route_after_anomaly

    plan = [{"tool": "anomaly", "sub_question": "which subscription rose"},
            {"tool": "sql", "sub_question": "by how much"}]
    assert route_after_anomaly(_state(plan)) == "sql_tool"


def test_an_anomaly_only_plan_still_stops_at_the_answer():
    """Unlike semantic, the detectors return real rows, so an extra SQL call
    would cost a round trip and change nothing."""
    from ledgerlens.agent.graph import route_after_anomaly

    assert route_after_anomaly(_state([{"tool": "anomaly"}])) == "answer"


def test_retrieved_ids_reach_sql_even_through_an_anomaly_step():
    """The semantic edge promises retrieval always reaches SQL. A plan naming
    semantic and anomaly but no sql would otherwise break that promise on a
    path the semantic test never visits."""
    from ledgerlens.agent.graph import route_after_anomaly

    state = _state([{"tool": "semantic"}, {"tool": "anomaly"}],
                   [{"tool": "semantic", "transaction_ids": [1, 2], "rows": []}])
    assert route_after_anomaly(state) == "sql_tool"


def test_semantic_goes_through_anomaly_when_both_are_planned():
    from ledgerlens.agent.graph import route_after_semantic

    plan = [{"tool": "semantic"}, {"tool": "anomaly"}]
    assert route_after_semantic(_state(plan)) == "anomaly_tool"
    assert route_after_semantic(_state([{"tool": "semantic"}])) == "sql_tool"


# --- how a retrieval-only result is described --------------------------------

def _semantic(ids):
    return {"tool": "semantic", "sub_question": "gym membership",
            "transaction_ids": ids, "rows": [], "error": None}


def _sql(value):
    return {"tool": "sql", "sub_question": "total", "query": "SELECT 1",
            "rows": [{"total": value}], "error": None}


def test_a_successful_retrieval_is_not_described_as_finding_nothing():
    """§6.5 forbids this tool from carrying a figure, so its result is always
    empty — and the generic empty-result wording then accused it of failure.
    The answer to a correctly answered question opened with "no matching
    transactions" and went on to give the right total."""
    from ledgerlens.agent.graph import draft_answer

    state = draft_answer({"question": "gym?", "answerable": True,
                          "tool_results": [_semantic([1, 2, 3]), _sql(-324.87)]})
    assert "no matching transactions" not in state["draft_answer"]
    assert "-324.87" in state["draft_answer"]


def test_retrieval_that_matched_nothing_still_says_so():
    from ledgerlens.agent.graph import draft_answer

    state = draft_answer({"question": "q", "answerable": True,
                          "tool_results": [_semantic([])]})
    assert "matched nothing" in state["draft_answer"]


def test_an_empty_semantic_result_does_not_make_a_full_answer_no_data():
    """`no_data` drives the badge beside the answer. Counting a tool that never
    returns rows would flag every retrieval-backed answer as empty."""
    from ledgerlens.agent.graph import draft_answer

    state = draft_answer({"question": "q", "answerable": True,
                          "tool_results": [_semantic([1, 2]), _sql(-324.87)]})
    assert not state["no_data"]


def test_a_genuinely_empty_sql_result_is_still_no_data():
    from ledgerlens.agent.graph import draft_answer

    state = draft_answer({"question": "q", "answerable": True,
                          "tool_results": [_semantic([1]), _sql(None)]})
    assert state["no_data"]
