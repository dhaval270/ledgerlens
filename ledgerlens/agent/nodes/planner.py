"""Planner node — §6.3.

In:  question + schema DDL + available tools + current date.
Out: strict JSON list of sub-questions, each tagged with a tool and a one-line
     rationale. No prose, no answer.

Routing guidance belongs in the prompt:
  countable / summable / comparable      → sql
  free-text recall ("that trip to Boston") → semantic
  "is this unusual", "did anything change" → anomaly

Multi-part questions must decompose properly: "Did I spend more on food than
last semester?" is two SQL calls plus a comparison, not one query.

On replan, verifier_verdict["reason"] is appended as extra context.
"""

from __future__ import annotations

from ..state import AgentState

MAX_REPLANS = 2


def planner(state: AgentState) -> AgentState:
    raise NotImplementedError
