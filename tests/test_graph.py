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
    assert out["retrieval_failed"] is True


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


def test_an_empty_result_is_neither_verified_nor_a_failure():
    """`SUM(amount) WHERE type='income'` over a ledger with none returns one
    NULL row. Observed live: rendered as "-> None" and badged verified."""
    out = draft_answer({
        "question": "How much money came in?",
        "answerable": True,
        "tool_results": [{"tool": "sql", "sub_question": "total income",
                          "rows": [{"SUM(amount)": None}], "error": None}],
    })
    assert out["no_data"] is True
    assert "None" not in out["draft_answer"]
    assert "no matching transactions" in out["draft_answer"]

    final = finalize(out | {"verifier_verdict": {"pass": True,
                                                 "reason": "answer makes no numeric claims"}})
    assert final["verifier_verdict"]["pass"] is False
    assert final["verifier_verdict"]["no_data"] is True
    # Correct answers do not get a caution prefix.
    assert "couldn't verify" not in final["draft_answer"]


def test_zero_rows_is_also_no_data():
    out = draft_answer({
        "question": "q", "answerable": True,
        "tool_results": [{"tool": "sql", "rows": [], "error": None}],
    })
    assert out["no_data"] is True


def test_a_real_value_is_not_no_data():
    out = draft_answer({
        "question": "q", "answerable": True,
        "tool_results": [{"tool": "sql", "rows": [{"total": -412.55}], "error": None}],
    })
    assert out["no_data"] is False
    assert "-412.55" in out["draft_answer"]


def test_zero_is_a_value_not_an_absence():
    """0.0 is falsy in Python and is a perfectly good answer."""
    out = draft_answer({
        "question": "q", "answerable": True,
        "tool_results": [{"tool": "sql", "rows": [{"total": 0.0}], "error": None}],
    })
    assert out["no_data"] is False


def test_a_retrieval_failure_is_never_reported_as_verified():
    """The verifier passes it trivially — it contains no figures to trace.

    Observed live: the SQL tool exhausted its attempts on a 429, the answer said
    "I couldn't answer that", and the API returned verified=true beside it. True
    in the letter, and precisely the pairing §6.7 exists to prevent.
    """
    out = finalize({
        "answerable": True,
        "retrieval_failed": True,
        "draft_answer": "I couldn't answer that — the query failed after repeated attempts.",
        "verifier_verdict": {"pass": True, "reason": "answer makes no numeric claims"},
    })
    assert out["verifier_verdict"]["pass"] is False
    # The answer already says it failed; saying so twice reads as two problems.
    assert "couldn't verify" not in out["draft_answer"]


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
