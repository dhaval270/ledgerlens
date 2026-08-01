"""Graph state — §6.1."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    question: str
    plan: list[dict]          # [{"sub_question", "tool", "rationale"}]
    tool_results: list[dict]  # [{"tool", "query", "rows", "error"}]
    draft_answer: str
    verifier_verdict: dict    # {"pass": bool, "reason": str}
    replan_count: int
    sql_attempts: int
