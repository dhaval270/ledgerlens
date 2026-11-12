"""StateGraph wiring — §6.2.

    question → planner
    planner  → semantic_tool → anomaly_tool → sql_tool → answer
               (each hop conditional; a plan naming one tool runs one tool)
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

from .nodes.anomaly_tool import anomaly_tool
from .nodes.planner import MAX_REPLANS, planner
from .nodes.semantic_tool import semantic_tool
from .nodes.sql_tool import sql_tool
from .nodes.verifier import verifier
from .state import AgentState


def route_after_planner(state: AgentState) -> str:
    """Refusals skip the tools entirely; otherwise dispatch by planned tool.

    Semantic runs before SQL when both are planned: §6.5 has retrieval produce
    transaction IDs that SQL then aggregates, so the order is a data dependency,
    not a preference.
    """
    if not state.get("answerable", True) or not state.get("plan"):
        return "answer"

    tools = {step.get("tool") for step in state["plan"]}
    if "semantic" in tools:
        return "semantic_tool"
    if "sql" in tools:
        return "sql_tool"
    if "anomaly" in tools:
        return "anomaly_tool"
    return "answer"


def _planned(state: AgentState, tool: str) -> bool:
    return any(step.get("tool") == tool for step in state.get("plan") or [])


def _has_retrieved_ids(state: AgentState) -> bool:
    return any(r.get("transaction_ids") for r in state.get("tool_results") or [])


def route_after_semantic(state: AgentState) -> str:
    """Always hand retrieved IDs to SQL. Retrieval alone cannot answer anything.

    This used to be conditional on the plan naming an `sql` step. But §6.5 has
    `semantic_tool` return transaction IDs and an explicitly empty `rows` — it
    is forbidden from producing a figure — so a semantic-only plan reached the
    answer node with nothing to render and reported "no matching transactions"
    over a retrieval that had in fact succeeded.

    The dead end was unreachable from the old eval, which called the SQL tool
    directly and never built the graph. It appeared the moment questions were
    scored end to end: three of ten semantic queries planned semantic-only and
    all three returned no_data despite retrieving the right rows.

    Routing unconditionally is safe because `sql_tool` falls back to the
    original question when the plan named no SQL step, and scopes itself to the
    retrieved IDs either way.

    Anomaly comes first when the plan asks for both, so a plan naming all three
    tools still runs all three.
    """
    return "anomaly_tool" if _planned(state, "anomaly") else "sql_tool"


def route_after_anomaly(state: AgentState) -> str:
    """Continue to SQL when the plan asked for it, or when IDs are waiting.

    The graph used to send every anomaly result straight to the answer node, so
    a plan naming both tools ran only the first. "Did any of my subscriptions
    go up in price?" plans anomaly (which charge moved) plus sql (by how much)
    and reached the answer with the sql step silently dropped — the plan was
    right, the run was short, and nothing reported a difference between them.
    It is scored as a routing failure by the eval precisely because the eval
    reads `tool_results` rather than the plan.

    Conditional, unlike the semantic edge: the detectors return real rows, so
    an anomaly-only question is already answerable and an unconditional hop to
    SQL would add a call that changes nothing.
    """
    if _planned(state, "sql") or _has_retrieved_ids(state):
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


def _is_empty(result: dict) -> bool:
    """No rows, or rows whose every cell is NULL.

    `SELECT SUM(amount) WHERE type='income'` over a ledger with no income
    returns one row containing NULL. That is a successful query that found
    nothing, not a value — rendering it as "None" invited reading a missing
    answer as an answer.
    """
    rows = result.get("rows") or []
    if not rows:
        return True
    return all(value is None for row in rows for value in row.values())


def _format_rows(result: dict) -> str:
    # Semantic results are empty by design — §6.5 forbids them from carrying a
    # figure — so the generic empty-result wording accused a successful
    # retrieval of having found nothing. The answer to "what do I pay for my
    # gym membership?" opened with "no matching transactions" and then gave the
    # correct total, which reads as a contradiction and is really a category
    # error about what this tool returns.
    if result.get("tool") == "semantic":
        found = len(result.get("transaction_ids") or [])
        return f"matched {found} transaction(s), totalled below" if found \
            else "matched nothing"

    if _is_empty(result):
        return "no matching transactions"
    rows = result["rows"]
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
        return {**state, "retrieval_failed": True,
                "draft_answer": "I couldn't retrieve anything for that question."}

    errored = [r for r in results if r.get("error")]
    if errored and not any(r.get("rows") for r in results):
        return {**state, "retrieval_failed": True, "draft_answer":
                "I couldn't answer that — the query failed after repeated attempts."}

    parts = [
        f"{r.get('sub_question') or state['question']} -> {_format_rows(r)}"
        for r in results
    ]
    # Semantic results are excluded from the no-data test for the same reason:
    # they are always empty, so counting them would report a fully answered
    # question as having matched nothing whenever retrieval ran.
    computed = [r for r in results if r.get("tool") != "semantic"]
    return {
        **state,
        "draft_answer": " | ".join(parts),
        "no_data": bool(computed) and all(_is_empty(r) for r in computed),
    }


def finalize(state: AgentState) -> AgentState:
    """Attach the verdict to the answer. An unverified figure is never presented bare."""
    verdict = state.get("verifier_verdict") or {}
    answer = state.get("draft_answer", "")

    # A retrieval failure is not a passed verification. The answer already says
    # it failed, so it is not re-prefixed — only the verdict is corrected, so
    # callers reading `verified` are not told a failure was checked and cleared.
    if state.get("retrieval_failed"):
        return {**state, "verifier_verdict": {
            "pass": False, "reason": "no tool result to verify against",
            "checked": 0, "unsupported": [],
        }}

    # An empty result is honest, so it gets no caution prefix — but it is not a
    # verified figure either, so `pass` stays false and `no_data` says why.
    if state.get("no_data"):
        return {**state, "verifier_verdict": {
            "pass": False, "no_data": True, "checked": 0, "unsupported": [],
            "reason": "the query ran and matched no rows — nothing to verify",
        }}

    if state.get("answerable", True) and verdict and not verdict.get("pass"):
        answer = (
            f"I couldn't verify this answer, so treat it with caution. "
            f"({verdict.get('reason', 'verification failed')}) {answer}"
        )

    return {**state, "draft_answer": answer}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("semantic_tool", semantic_tool)
    graph.add_node("anomaly_tool", anomaly_tool)
    graph.add_node("sql_tool", sql_tool)
    graph.add_node("answer", draft_answer)
    graph.add_node("verifier", verifier)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", route_after_planner,
                                {"semantic_tool": "semantic_tool",
                                 "sql_tool": "sql_tool",
                                 "anomaly_tool": "anomaly_tool",
                                 "answer": "answer"})
    graph.add_conditional_edges("semantic_tool", route_after_semantic,
                                {"sql_tool": "sql_tool",
                                 "anomaly_tool": "anomaly_tool"})
    graph.add_conditional_edges("anomaly_tool", route_after_anomaly,
                                {"sql_tool": "sql_tool", "answer": "answer"})
    graph.add_edge("sql_tool", "answer")
    graph.add_edge("answer", "verifier")
    graph.add_conditional_edges("verifier", route_after_verifier,
                                {"planner": "planner", "answer": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile()


def ask(question: str, thread_id: str | None = None) -> dict:
    """Run one question end to end.

    With a `thread_id`, prior turns are shown to the planner so a follow-up can
    be resolved ("and in May?"). Without one, the question stands alone — which
    is what every eval does deliberately, so a benchmark score cannot be
    inflated by a neighbouring question in the file.
    """
    from .memory import history, remember

    final = build_graph().invoke({
        "question": question,
        "plan": [],
        "tool_results": [],
        "replan_count": 0,
        "sql_attempts": 0,
        "history": history(thread_id),
    })
    verdict = final.get("verifier_verdict") or {}
    answer = final.get("draft_answer", "")
    remember(thread_id, question, answer)
    return {
        "answer": answer,
        "verified": bool(verdict.get("pass")),
        "no_data": bool(final.get("no_data")),
        "verdict": verdict,
        "plan": final.get("plan", []),
        "tool_results": final.get("tool_results", []),
        "answerable": final.get("answerable", True),
        "thread_id": thread_id,
    }
