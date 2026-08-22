"""Planner tests — §6.3.

The LLM call is stubbed. What is tested here is the logic around it: plan
trimming, refusal handling, and the replan bookkeeping that decides whether the
graph loops or gives up.
"""

from __future__ import annotations

import pytest

from ledgerlens.agent.nodes import planner as planner_mod
from ledgerlens.agent.nodes.planner import MAX_STEPS, Plan, Step, Tool, planner


def make(monkeypatch, plan: Plan):
    monkeypatch.setattr(planner_mod, "make_plan",
                        lambda question, feedback="", history=None: plan)


def step(n: int = 1, tool: Tool = Tool.SQL) -> Step:
    return Step(sub_question=f"q{n}", tool=tool, rationale="because")


# --- refusal -----------------------------------------------------------------

def test_unanswerable_question_produces_no_steps(monkeypatch):
    make(monkeypatch, Plan(answerable=False, refusal_reason="no forecasting", steps=[]))
    out = planner({"question": "What will I spend next month?"})
    assert out["answerable"] is False
    assert out["plan"] == []
    assert "forecast" in out["refusal_reason"]


def test_refusal_discards_any_steps_the_model_supplied(monkeypatch):
    """A model that says 'unanswerable' but still routes work must not run it."""
    monkeypatch.setattr(
        planner_mod, "get_structured_llm", lambda schema: None, raising=False
    )
    plan = Plan(answerable=False, refusal_reason="not in the data", steps=[step()])
    make(monkeypatch, plan)
    assert planner({"question": "What is my credit score?"})["plan"] == []


# --- decomposition -----------------------------------------------------------

def test_simple_question_keeps_a_single_step(monkeypatch):
    make(monkeypatch, Plan(answerable=True, steps=[step()]))
    out = planner({"question": "How much did I spend on groceries in March?"})
    assert len(out["plan"]) == 1
    assert out["plan"][0]["tool"] == "sql"


def test_plan_is_serialised_as_plain_dicts(monkeypatch):
    """AgentState carries JSON-able values; enums must not leak into it."""
    make(monkeypatch, Plan(answerable=True, steps=[step(tool=Tool.ANOMALY)]))
    entry = planner({"question": "Is my spending unusual?"})["plan"][0]
    assert entry["tool"] == "anomaly"
    assert isinstance(entry["tool"], str)


def test_multi_step_plan_is_preserved(monkeypatch):
    make(monkeypatch, Plan(answerable=True, steps=[step(1), step(2)]))
    out = planner({"question": "Did I spend more on food this year than last?"})
    assert len(out["plan"]) == 2


# --- replan bookkeeping ------------------------------------------------------

def test_first_pass_does_not_count_as_a_replan(monkeypatch):
    make(monkeypatch, Plan(answerable=True, steps=[step()]))
    assert planner({"question": "q"})["replan_count"] == 0


def test_failed_verification_increments_replan_count(monkeypatch):
    make(monkeypatch, Plan(answerable=True, steps=[step()]))
    out = planner({
        "question": "q",
        "verifier_verdict": {"pass": False, "reason": "figure not found"},
        "replan_count": 0,
    })
    assert out["replan_count"] == 1


def test_passing_verdict_does_not_increment(monkeypatch):
    make(monkeypatch, Plan(answerable=True, steps=[step()]))
    out = planner({
        "question": "q",
        "verifier_verdict": {"pass": True, "reason": "all figures trace"},
        "replan_count": 1,
    })
    assert out["replan_count"] == 1


def test_verifier_reason_is_fed_back_into_planning(monkeypatch):
    """§6.7: on failure the reason becomes extra planner context."""
    seen = {}

    def capture(question, feedback="", history=None):
        seen["feedback"] = feedback
        return Plan(answerable=True, steps=[step()])

    monkeypatch.setattr(planner_mod, "make_plan", capture)
    planner({
        "question": "q",
        "verifier_verdict": {"pass": False, "reason": "figure 999.99 not found"},
        "replan_count": 0,
    })
    assert "999.99" in seen["feedback"]


# --- schema constraints ------------------------------------------------------

def test_tool_enum_rejects_an_invented_tool():
    with pytest.raises(ValueError):
        Step(sub_question="q", tool="spreadsheet", rationale="r")


def test_plan_defaults_are_safe():
    plan = Plan(answerable=True)
    assert plan.steps == []
    assert plan.refusal_reason == ""


def test_max_steps_is_small_enough_to_bound_cost():
    """Every extra step is another LLM call and more rows to verify."""
    assert MAX_STEPS <= 4
