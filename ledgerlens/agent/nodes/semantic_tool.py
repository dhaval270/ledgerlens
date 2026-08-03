"""Semantic recall — §6.5.

Embed `raw_descriptor + merchant + category + month` per transaction, retrieve
top-k, return transaction IDs **only**.

Those IDs are then handed to the SQL tool for any aggregation. The embedding
layer must never produce a number — that is non-negotiable #1. Nearest-neighbour
scores are similarities, not evidence: a transaction can rank first and still be
irrelevant, and summing retrieved rows in Python would put a figure in the
answer that no query ever computed and the verifier cannot trace.

**Deviation from §2.** The spec suggests `sqlite-vec` or FAISS. `sqlite-vec` is
installed but cannot load here: CPython's bundled `sqlite3` on macOS is compiled
without `enable_load_extension`, so the extension is unreachable. Similarity is
therefore computed with numpy over a normalized matrix. For 832 rows that is a
sub-millisecond dot product and still fully in-process, which is what §2 was
actually asking for. At a scale where this matters, the fix is Postgres with
pgvector — the same answer §14 gives for scaling generally.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from ...db import DB_PATH, connect_readonly
from ..state import AgentState

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 20

INDEX_PATH = DB_PATH.with_suffix(".vectors.npz")

_MODEL = None


def _model():
    """Loaded lazily: importing sentence-transformers costs seconds, and the
    deterministic tiers must stay usable without paying that."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(EMBED_MODEL)
    return _MODEL


def embedding_text(row: sqlite3.Row) -> str:
    """§6.5's four fields. Month is included so "that trip in April" has
    something to match on beyond the merchant string."""
    parts = [
        row["raw_descriptor"],
        row["merchant"] or "",
        row["category"] or "",
        (row["posted_date"] or "")[:7],
    ]
    return " | ".join(p for p in parts if p)


def build_index(db_path: Path | str = DB_PATH, index_path: Path = INDEX_PATH) -> int:
    """Embed every transaction and persist the matrix. Returns rows indexed."""
    conn = connect_readonly(db_path)
    try:
        rows = conn.execute(
            """SELECT t.id, t.raw_descriptor, t.posted_date,
                      m.canonical_name AS merchant, c.name AS category
               FROM transactions t
               LEFT JOIN merchants m ON m.id = t.merchant_id
               LEFT JOIN categories c ON c.id = t.category_id
               ORDER BY t.id"""
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return 0

    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    vectors = _model().encode(
        [embedding_text(r) for r in rows],
        normalize_embeddings=True,      # cosine similarity becomes a dot product
        show_progress_bar=False,
    ).astype(np.float32)

    np.savez(index_path, ids=ids, vectors=vectors)
    return len(ids)


def _load_index(index_path: Path = INDEX_PATH):
    if not index_path.exists():
        raise FileNotFoundError(
            f"No semantic index at {index_path}. Run: python -m ledgerlens.index"
        )
    data = np.load(index_path)
    return data["ids"], data["vectors"]


def search(query: str, k: int = TOP_K, index_path: Path = INDEX_PATH) -> list[int]:
    """Top-k transaction IDs by cosine similarity. Returns IDs, never values."""
    ids, vectors = _load_index(index_path)
    q = _model().encode([query], normalize_embeddings=True).astype(np.float32)[0]

    scores = vectors @ q
    k = min(k, len(ids))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [int(ids[i]) for i in top]


def semantic_tool(state: AgentState) -> AgentState:
    """Returns {"tool": "semantic", "transaction_ids": [...]} — never aggregates."""
    results = list(state.get("tool_results", []))

    targets = [s for s in state.get("plan", []) if s.get("tool") == "semantic"]
    if not targets:
        targets = [{"sub_question": state["question"]}]

    for step in targets:
        question = step.get("sub_question", state["question"])
        try:
            ids = search(question)
            entry = {
                "tool": "semantic",
                "sub_question": question,
                "transaction_ids": ids,
                # Deliberately empty: the verifier reconciles numbers against
                # rows, and this tool must not supply any.
                "rows": [],
                "error": None,
            }
        except Exception as exc:
            entry = {
                "tool": "semantic",
                "sub_question": question,
                "transaction_ids": [],
                "rows": [],
                "error": f"semantic search failed: {exc}",
            }
        results.append(entry)

    return {**state, "tool_results": results}
