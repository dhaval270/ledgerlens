"""Semantic recall — §6.5.

Embed `raw_descriptor + merchant + category + month` per transaction, retrieve
top-k, return transaction IDs only.

Those IDs are then handed to the SQL tool for any aggregation. The embedding
layer must never produce a number — that is non-negotiable #1.

Embeddings stay local (sentence-transformers/all-MiniLM-L6-v2); Groq serves chat
models only.
"""

from __future__ import annotations

from ..state import AgentState

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 20


def semantic_tool(state: AgentState) -> AgentState:
    """Returns {"tool": "semantic", "transaction_ids": [...]} — never aggregates."""
    raise NotImplementedError
