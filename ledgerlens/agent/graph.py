"""StateGraph wiring — §6.2.

    question → planner
    planner  → [sql_tool | semantic_tool | anomaly_tool]   (conditional, fan-out)
    tools    → verifier
    verifier → planner  (on fail, replan_count < 2)
    verifier → answer   (on pass)
    verifier → answer   (on fail + replan_count == 2, honest "couldn't verify")

The answer node composes prose from tool results and never computes: every
figure it prints must already exist in a row, or the verifier rejects it. That
is non-negotiable #1 expressed as control flow rather than as an instruction.

The two-replan cap matters. Without it a verifier that cannot be satisfied loops
forever, and each cycle costs a full planner-plus-tools round trip. Hitting the
cap is a legitimate outcome: §6.7 says an honest "I couldn't verify this" beats
a confident wrong number.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes.planner import MAX_REPLANS, planner
from .nodes.sql_tool import sql_tool
from .nodes.verifier import verifier
from .state import AgentState


def route_after_planner(state: AgentState) -> str:
    """Refusals skip the tools entirely; otherwise fan out by planned tool."""
    if not state.get("answerable", True):
        return "answer"
    if not state.get("plan"):
        return "answer"

    tools = {step.get("tool") for step in state["plan"]}
    # Only the SQL tool is implemented; semantic and anomaly are Week 3 (§11).
    if "sql" in tools:
        return "sql_tool"
    return "answer"


def route_after_verifier(state: AgentState) -> str:
    """Replan while the budget allows, otherwise answer honestly."""
    verdict = state.get("verifier_verdict") or {}
    if verdict.get("pass"):
        return "answer"
    if state.get("replan_count", 0) < MAX_REPLANS:
        return "planner"
    return "answer"


def _format_rows(result: dict) -> str:
    rows = result.get("rows") or []
    if not rows:
        return "no rows"
    if len(rows) == 1 and len(rows[0]) == 1:
        return str(next(iter(rows[0].values())))
    return "; ".join(", ".join(f"{k}={v}" for k, v in row.items()) for row in rows[:5])


def draft_answer(state: AgentState) -> AgentState:
    """Compose the answer from retrieved rows. Deterministic, no LLM, no maths.

    Kept as string assembly on purpose: anything this node invents is a figure
    with no provenance, and the verifier would reject it anyway. Narration is a
    later concern (§7's digest), correctness is this one.
    """
    if not state.get("answerable", True):
        reason = state.get("refusal_reason") or "this cannot be answered from the ledger"
        return {**state, "draft_answer": f"I can't answer that: {reason}"}

    results = state.get("tool_results") or []
    if not results:
        return {**state, "draft_answer": "I couldn't retrieve anything for that question."}

    errored = [r for r in results if r.get("error")]
    if errored and not any(r.get("rows") for r in results):
        return {**state, "draft_answer":
                "I couldn't answer that — the query failed after repeated attempts."}

    parts = [
        f"{r.get('sub_question') or state['question']} -> {_format_rows(r)}"
        for r in results
    ]
    return {**state, "draft_answer": " | ".join(parts)}


def finalize(state: AgentState) -> AgentState:
    """Attach the verdict to the answer. An unverified figure is never presented bare."""
    verdict = state.get("verifier_verdict") or {}
    answer = state.get("draft_answer", "")

    if state.get("answerable", True) and verdict and not verdict.get("pass"):
        answer = (
            f"I couldn't verify this answer, so treat it with caution. "
            f"({verdict.get('reason', 'verification failed')}) {answer}"
        )

    return {**state, "draft_answer": answer}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("sql_tool", sql_tool)
    graph.add_node("answer", draft_answer)
    graph.add_node("verifier", verifier)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", route_after_planner,
                                {"sql_tool": "sql_tool", "answer": "answer"})
    graph.add_edge("sql_tool", "answer")
    graph.add_edge("answer", "verifier")
    graph.add_conditional_edges("verifier", route_after_verifier,
                                {"planner": "planner", "answer": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile()


def ask(question: str) -> dict:
    """Run one question end to end."""
    final = build_graph().invoke({
        "question": question,
        "plan": [],
        "tool_results": [],
        "replan_count": 0,
        "sql_attempts": 0,
    })
    verdict = final.get("verifier_verdict") or {}
    return {
        "answer": final.get("draft_answer", ""),
        "verified": bool(verdict.get("pass")),
        "verdict": verdict,
        "plan": final.get("plan", []),
        "tool_results": final.get("tool_results", []),
        "answerable": final.get("answerable", True),
    }
