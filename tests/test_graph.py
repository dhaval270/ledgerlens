"""Graph wiring and routing — §6.2. No network: routing is pure logic."""

from __future__ import annotations

from ledgerlens.agent.graph import (
    build_graph,
    draft_answer,
    finalize,
    route_after_planner,
    route_after_verifier,
)
from ledgerlens.agent.nodes.planner import MAX_REPLANS


# --- routing after the planner ----------------------------------------------

def test_refusal_skips_the_tools():
    state = {"answerable": False, "refusal_reason": "no forecasting", "plan": []}
    assert route_after_planner(state) == "answer"


def test_sql_plan_routes_to_the_sql_tool():
    state = {"answerable": True, "plan": [{"tool": "sql", "sub_question": "q"}]}
    assert route_after_planner(state) == "sql_tool"


def test_empty_plan_routes_straight_to_answer():
    assert route_after_planner({"answerable": True, "plan": []}) == "answer"


# --- routing after the verifier ---------------------------------------------

def test_passing_verdict_goes_to_answer():
    state = {"verifier_verdict": {"pass": True}, "replan_count": 0}
    assert route_after_verifier(state) == "answer"


def test_failed_verdict_replans_while_budget_remains():
    state = {"verifier_verdict": {"pass": False}, "replan_count": 0}
    assert route_after_verifier(state) == "planner"


def test_replan_budget_is_finite():
    """Without a cap an unsatisfiable verifier loops forever."""
    state = {"verifier_verdict": {"pass": False}, "replan_count": MAX_REPLANS}
    assert route_after_verifier(state) == "answer"


# --- answer composition ------------------------------------------------------

def test_refusal_answer_states_the_reason():
    out = draft_answer({"question": "What will I spend next month?",
                        "answerable": False, "refusal_reason": "no forecasting"})
    assert "can't answer" in out["draft_answer"]
    assert "forecasting" in out["draft_answer"]


def test_answer_reports_retrieved_values():
    out = draft_answer({
        "question": "How much on groceries?",
        "answerable": True,
        "tool_results": [{"tool": "sql", "sub_question": "How much on groceries?",
                          "rows": [{"total": -412.55}], "error": None}],
    })
    assert "-412.55" in out["draft_answer"]


def test_total_tool_failure_is_reported_not_papered_over():
    out = draft_answer({
        "question": "q", "answerable": True,
        "tool_results": [{"tool": "sql", "rows": [], "error": "OperationalError"}],
    })
    assert "couldn't answer" in out["draft_answer"]


# --- finalize ----------------------------------------------------------------

def test_unverified_answer_is_flagged_to_the_reader():
    """§6.7: an honest 'I couldn't verify this' beats a confident wrong number."""
    out = finalize({
        "answerable": True,
        "draft_answer": "You spent $500.00.",
        "verifier_verdict": {"pass": False, "reason": "figure not found in any tool result"},
    })
    assert "couldn't verify" in out["draft_answer"]
    assert "figure not found" in out["draft_answer"]


def test_verified_answer_is_left_alone():
    out = finalize({
        "answerable": True,
        "draft_answer": "You spent $412.55.",
        "verifier_verdict": {"pass": True, "reason": "all figures trace"},
    })
    assert out["draft_answer"] == "You spent $412.55."


def test_refusal_is_not_labelled_unverified():
    """A refusal is a correct answer, not a failed one."""
    out = finalize({
        "answerable": False,
        "draft_answer": "I can't answer that: no forecasting.",
        "verifier_verdict": {"pass": False, "reason": "no tool results to verify against"},
    })
    assert "couldn't verify" not in out["draft_answer"]


# --- compilation -------------------------------------------------------------

def test_graph_compiles():
    assert build_graph() is not None
