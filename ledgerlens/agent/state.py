"""Graph state — §6.1."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    question: str
    plan: list[dict]          # [{"sub_question", "tool", "rationale"}]
    tool_results: list[dict]  # [{"tool", "query", "rows", "error"}]
    draft_answer: str
    verifier_verdict: dict    # {"pass": bool, "reason": str}
    replan_count: int
    sql_attempts: int

    # Prior turns of this conversation, oldest first, as {"question", "answer"}.
    # Read by the planner and by nothing else — see agent/memory.py for why a
    # figure must never travel this way.
    history: list[dict]

    # Set by the planner. A question the ledger cannot answer is refused before
    # any tool runs — §9.1 makes refusal the correct answer for 7 golden
    # queries, and routing them anyway invites a tool to invent a number.
    answerable: bool
    refusal_reason: str

    # Set by the answer node when no tool produced a row. The verifier passes
    # such an answer trivially — it contains no figures, so nothing fails to
    # trace — and "I couldn't answer that" would then be reported as verified.
    # Technically true, and exactly the misleading pairing §6.7 exists to
    # prevent, so the failure is carried explicitly rather than inferred from a
    # verdict that was never about retrieval.
    retrieval_failed: bool

    # Set by the answer node when the tools ran fine and matched nothing. This
    # is a third outcome, not a shade of the other two: "you have no income" is
    # a correct answer, so it must not be flagged as a failure — but it verifies
    # trivially (no figures, nothing to trace), so a green tick beside it claims
    # a check that never happened. Both readings are wrong; it needs its own.
    no_data: bool
