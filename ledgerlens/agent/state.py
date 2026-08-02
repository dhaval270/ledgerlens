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

    # Set by the planner. A question the ledger cannot answer is refused before
    # any tool runs — §9.1 makes refusal the correct answer for 7 golden
    # queries, and routing them anyway invites a tool to invent a number.
    answerable: bool
    refusal_reason: str
