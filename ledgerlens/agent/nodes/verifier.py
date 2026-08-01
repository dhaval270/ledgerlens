"""Verifier — §6.7. The differentiating node.

Given draft_answer and tool_results:
  1. extract every numeric literal in the draft
  2. check each against values present in tool_results rows (rounding tolerance)
  3. check no claim asserts anything outside the retrieved date range
  4. return {"pass": bool, "reason": str}

Steps 1-3 are code, not LLM — that is the whole point. Only the "reason"
phrasing is generated. On failure the reason feeds back into the planner.

After two failures the answer says so plainly. An honest "I couldn't verify
this" beats a confident wrong number.
"""

from __future__ import annotations

from ..state import AgentState

ROUNDING_TOLERANCE = 0.01


def extract_numerics(text: str) -> list[float]:
    """Every numeric literal in the draft answer, currency symbols stripped."""
    raise NotImplementedError


def verifier(state: AgentState) -> AgentState:
    raise NotImplementedError
