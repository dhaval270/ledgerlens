"""Approval graph — §8.

    propose → review ──approve──→ apply → END
                     └──reject──→ cancel → END

`review` calls LangGraph's `interrupt`, so the run genuinely stops mid-graph:
the checkpointer holds the pending proposal and nothing after that node has
executed. The write is not queued, guarded or flagged — it has not happened,
and it cannot happen until someone resumes the thread with an approval.

That is why the interrupt sits *between* preview and apply rather than around a
single write function. A function that decides and writes can always be called
with the decision defaulted; a graph paused at an interrupt has no default to
default to. The pause is the mechanism, not a check.

`propose` runs read-only. `apply` is the only place in the agent path that
opens a writable connection.
"""

from __future__ import annotations

import functools
import itertools
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from ..db import DB_PATH, connect, connect_readonly
from .actions import ApprovalError, StaleProposal, build_proposal, commit_proposal

_thread_ids = itertools.count(1)

# A node cannot raise through a checkpointed graph without leaving the thread
# mid-step, so failures travel as state. The class travels with the message:
# "the ledger moved" and "that category does not exist" call for different
# handling, and collapsing both to ApprovalError would hide that from callers.
_ERRORS = {cls.__name__: cls for cls in (ApprovalError, StaleProposal)}


def _raise(state: "ApprovalState") -> None:
    if state.get("error"):
        raise _ERRORS.get(state.get("error_type", ""), ApprovalError)(state["error"])


class ApprovalState(TypedDict, total=False):
    action: str
    params: dict
    db_path: str
    proposal: dict | None
    approved: bool
    note: str
    result: dict | None
    error: str | None
    error_type: str | None


def _propose_node(state: ApprovalState) -> ApprovalState:
    """Describe the change. Read-only connection: a preview cannot write."""
    conn = connect_readonly(state.get("db_path") or DB_PATH)
    try:
        proposal = build_proposal(conn, state["action"], state.get("params") or {})
    except ApprovalError as exc:
        return {**state, "proposal": None, "error": str(exc),
                "error_type": type(exc).__name__}
    finally:
        conn.close()
    return {**state, "proposal": proposal.as_dict(), "error": None, "error_type": None}


def _review_node(state: ApprovalState) -> ApprovalState:
    """Stop here and wait for a human.

    The interrupt payload is the diff itself, so whatever resumes the thread —
    CLI, API, notebook — shows the user the same before/after that `apply` will
    verify against.
    """
    decision = interrupt({"pending": state["proposal"]})

    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        note = str(decision.get("note", ""))
    else:
        approved, note = bool(decision), ""
    return {**state, "approved": approved, "note": note}


def _apply_node(state: ApprovalState) -> ApprovalState:
    conn = connect(state.get("db_path") or DB_PATH)
    try:
        result = commit_proposal(conn, state["proposal"])
    except ApprovalError as exc:
        return {**state, "result": None, "error": str(exc),
                "error_type": type(exc).__name__}
    finally:
        conn.close()
    return {**state, "result": result, "error": None, "error_type": None}


def _cancel_node(state: ApprovalState) -> ApprovalState:
    return {**state, "result": None, "error": None}


def _route_after_propose(state: ApprovalState) -> str:
    return "review" if state.get("proposal") else END


def _route_after_review(state: ApprovalState) -> str:
    return "apply" if state.get("approved") else "cancel"


@functools.lru_cache(maxsize=1)
def build_approval_graph():
    """Compiled once: the checkpointer must outlive a single propose/decide pair."""
    graph = StateGraph(ApprovalState)
    graph.add_node("propose", _propose_node)
    graph.add_node("review", _review_node)
    graph.add_node("apply", _apply_node)
    graph.add_node("cancel", _cancel_node)

    graph.set_entry_point("propose")
    graph.add_conditional_edges("propose", _route_after_propose,
                                {"review": "review", END: END})
    graph.add_conditional_edges("review", _route_after_review,
                                {"apply": "apply", "cancel": "cancel"})
    graph.add_edge("apply", END)
    graph.add_edge("cancel", END)

    return graph.compile(checkpointer=MemorySaver())


# --- facade ------------------------------------------------------------------

def propose(action: str, thread_id: str | None = None,
            db_path: str | None = None, **params: Any) -> dict:
    """Start an approval. Returns the pending diff; writes nothing.

    Raises ApprovalError when the change is impossible or already true — a
    proposal that cannot be applied should fail here, in front of the user,
    rather than after they approve it.
    """
    thread_id = thread_id or f"approval-{next(_thread_ids)}"
    state = build_approval_graph().invoke(
        {"action": action, "params": params, "db_path": str(db_path or DB_PATH)},
        config={"configurable": {"thread_id": thread_id}},
    )
    _raise(state)
    return {"thread_id": thread_id, "proposal": state["proposal"], "status": "pending"}


def pending(thread_id: str) -> dict | None:
    """The diff a paused thread is waiting on, or None if it is not paused."""
    snapshot = build_approval_graph().get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.next:
        return None
    return snapshot.values.get("proposal")


def decide(thread_id: str, approved: bool, note: str = "") -> dict:
    """Resume a paused approval. This is the only path to a write."""
    state = build_approval_graph().invoke(
        Command(resume={"approved": approved, "note": note}),
        config={"configurable": {"thread_id": thread_id}},
    )
    _raise(state)
    return {
        "thread_id": thread_id,
        "status": "applied" if approved else "rejected",
        "proposal": state.get("proposal"),
        "result": state.get("result"),
    }
