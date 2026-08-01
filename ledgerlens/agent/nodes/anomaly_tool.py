"""Statistical anomaly detection — §6.6.

Per category, compute a rolling baseline from prior months and flag via z-score
or IQR. Returns structured findings, not prose.
"""

from __future__ import annotations

from ..state import AgentState

Z_THRESHOLD = 2.0
BASELINE_MONTHS = 6


def anomaly_tool(state: AgentState) -> AgentState:
    """Returns structured findings: category, observed, baseline, z, direction."""
    raise NotImplementedError
