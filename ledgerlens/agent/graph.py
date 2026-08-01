"""StateGraph wiring — §6.2.

    question → planner
    planner  → [sql_tool | semantic_tool | anomaly_tool]   (conditional, fan-out)
    tools    → verifier
    verifier → planner  (on fail, replan_count < 2)
    verifier → answer   (on pass)
    verifier → answer   (on fail + replan_count == 2, emitting an honest
                         "couldn't verify")

§8 routes any proposed state change through interrupt() and waits for approval.
"""

from __future__ import annotations

from langgraph.graph import StateGraph

from .state import AgentState


def route_after_planner(state: AgentState) -> list[str]:
    """Fan out to whichever tools the plan named."""
    raise NotImplementedError


def route_after_verifier(state: AgentState) -> str:
    """'planner' while replan_count < 2, otherwise 'answer'."""
    raise NotImplementedError


def build_graph() -> StateGraph:
    raise NotImplementedError
