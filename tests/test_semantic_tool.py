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


def test_semantic_only_plan_goes_straight_to_answer():
    assert route_after_semantic({"plan": [{"tool": "semantic"}]}) == "answer"
