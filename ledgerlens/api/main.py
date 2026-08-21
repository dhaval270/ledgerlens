"""FastAPI surface — §3.

Statements are uploaded manually, so /ingest is the primary entry point rather
than §5's watched folder. Uploads land in data/private/, which is gitignored
per §10.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..approvals import ApprovalError, StaleProposal, decide, pending, propose
from ..db import DB_PATH, connect, init_db
from ..ingest import ingest_file, resolve_pending
from ..ingest.parse import (
    EncryptedStatement,
    NoTextLayer,
    StatementError,
    UnknownFormat,
)

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "private"
UI_PATH = Path(__file__).resolve().parent / "ui.html"
ALLOWED_SUFFIXES = {".pdf", ".csv"}

app = FastAPI(title="LedgerLens", version="0.1.0")


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    verified: bool
    no_data: bool = False
    verdict: dict
    plan: list[dict]
    tool_results: list[dict]
    answerable: bool


class ProposalRequest(BaseModel):
    action: str
    params: dict = {}


class Decision(BaseModel):
    approved: bool
    note: str = ""


@app.get("/")
def root() -> dict:
    """Landing route — hitting / and getting a 404 reads as a broken server."""
    return {
        "service": "LedgerLens",
        "ui": "/ui",
        "docs": "/docs",
        "endpoints": {
            "GET /ui": "the operator page",
            "GET /health": "service and database status",
            "POST /ingest": "upload a statement (.csv/.pdf)",
            "POST /ask": "ask a question about the ledger",
            "GET /digest/{period}": "proactive findings for a YYYY-MM month",
            "POST /approvals": "propose a change — returns a diff, writes nothing",
            "GET /approvals/{thread_id}": "the diff a paused approval is waiting on",
            "POST /approvals/{thread_id}/decide": "approve or reject — the only path to a write",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "db": DB_PATH.exists()}


@app.get("/ui", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    """The operator page. Served from disk so editing it needs no restart.

    Explicitly uncacheable. Without these headers browsers heuristically cache a
    response that carries no `Cache-Control`, which produced the worst kind of
    confusion: fresh JSON from the API rendered by stale JavaScript, so half the
    page reflected an edit and half did not, and nothing looked broken enough to
    suspect the cache. "Edit and reload" has to actually mean that.
    """
    return HTMLResponse(
        UI_PATH.read_text(),
        headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/stats")
def stats() -> dict:
    """What is actually in the open database.

    Reports `db` first because the most confusing failure in this app is not an
    error at all: ingesting into one ledger and reading from another. The file
    on screen removes the guesswork.
    """
    _require_ledger()
    conn = connect()
    try:
        (total, first, last) = conn.execute(
            "SELECT COUNT(*), MIN(posted_date), MAX(posted_date) FROM transactions"
        ).fetchone()
        return {
            "db": str(DB_PATH),
            "transactions": total,
            "first_date": first,
            "last_date": last,
            "by_type": dict(conn.execute(
                "SELECT type, COUNT(*) FROM transactions GROUP BY type ORDER BY 2 DESC")),
            "by_source": dict(conn.execute(
                "SELECT source_file, COUNT(*) FROM transactions GROUP BY source_file "
                "ORDER BY 2 DESC")),
            "by_category": [
                {"category": r[0] or "uncategorized", "n": r[1], "total": round(r[2], 2)}
                for r in conn.execute(
                    """SELECT c.name, COUNT(*), SUM(t.amount) FROM transactions t
                       LEFT JOIN categories c ON c.id = t.category_id
                       WHERE t.type = 'purchase'
                       GROUP BY c.name ORDER BY SUM(t.amount)""")
            ],
            "tables": {
                name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                for (name,) in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name")
            },
        }
    finally:
        conn.close()


@app.get("/transactions")
def transactions(limit: int = 50, offset: int = 0, q: str = "",
                 type: str = "", source: str = "") -> dict:
    """Paged transaction list, newest first.

    Filters are bound parameters, never string-formatted into the SQL. This
    endpoint takes user text and runs it against the same database the agent
    reads, so it gets the same discipline as §6.4's generated queries.
    """
    _require_ledger()
    limit = max(1, min(limit, 500))

    where, params = ["1=1"], []
    if q:
        where.append("(t.raw_descriptor LIKE ? OR m.canonical_name LIKE ?)")
        params += [f"%{q}%", f"%{q}%"]
    if type:
        where.append("t.type = ?")
        params.append(type)
    if source:
        where.append("t.source_file = ?")
        params.append(source)
    clause = " AND ".join(where)

    conn = connect()
    try:
        joins = ("FROM transactions t "
                 "LEFT JOIN merchants m ON m.id = t.merchant_id "
                 "LEFT JOIN categories c ON c.id = t.category_id")
        (total,) = conn.execute(f"SELECT COUNT(*) {joins} WHERE {clause}", params).fetchone()
        (matched_sum,) = conn.execute(
            f"SELECT COALESCE(SUM(t.amount), 0) {joins} WHERE {clause}", params).fetchone()
        rows = conn.execute(
            f"""SELECT t.id, t.posted_date, t.amount, t.type, t.raw_descriptor,
                       m.canonical_name AS merchant, c.name AS category,
                       t.categorized_by, t.source_file
                {joins} WHERE {clause}
                ORDER BY t.posted_date DESC, t.id DESC LIMIT ? OFFSET ?""",
            [*params, limit, offset],
        ).fetchall()
        return {
            "total": total,
            "sum": round(matched_sum, 2),
            "limit": limit,
            "offset": offset,
            "rows": [dict(r) for r in rows],
        }
    finally:
        conn.close()


@app.get("/meta")
def meta() -> dict:
    """Categories and merchants, for the approval forms.

    The UI offers these as dropdowns rather than free-text ids: a proposal
    naming a category that does not exist is refused at propose time anyway, and
    making it unselectable is friendlier than making it an error.
    """
    _require_ledger()
    conn = connect()
    try:
        return {
            "categories": [r[0] for r in conn.execute(
                "SELECT name FROM categories ORDER BY name")],
            "merchants": [
                {"id": r[0], "name": r[1], "category": r[2], "transactions": r[3]}
                for r in conn.execute(
                    """SELECT m.id, m.canonical_name, c.name, COUNT(t.id)
                       FROM merchants m
                       LEFT JOIN categories c ON c.id = m.default_category_id
                       LEFT JOIN transactions t ON t.merchant_id = m.id
                       GROUP BY m.id ORDER BY m.canonical_name""")
            ],
            "periods": [r[0] for r in conn.execute(
                """SELECT DISTINCT substr(posted_date,1,7) FROM transactions
                   ORDER BY 1 DESC""")],
        }
    finally:
        conn.close()


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """Upload one statement. Re-uploading the same file is a safe no-op (§5.2)."""
    name = Path(file.filename or "").name  # strip any path components
    if not name:
        raise HTTPException(400, "No filename provided")
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"Unsupported file type. Allowed: {sorted(ALLOWED_SUFFIXES)}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / name
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    if not DB_PATH.exists():
        init_db()

    try:
        with connect() as conn:
            result = ingest_file(conn, dest)
            # Resolve here, not only in the CLI: uploads that skipped this step
            # landed with NULL merchant and category, so the rows were visible in
            # the table and unreachable by any query that joins `merchants`.
            resolution = resolve_pending(conn)
    except (EncryptedStatement, NoTextLayer, UnknownFormat) as exc:
        # Keep these distinct from a 500 — they are user-fixable input problems,
        # and the message says which one it is.
        raise HTTPException(422, str(exc)) from exc
    except StatementError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {**result.as_dict(), **resolution, **_reindex()}


def _reindex() -> dict:
    """Re-embed after an upload, so new rows are reachable by semantic search.

    The same omission as merchant resolution above, one layer out: §6.5's index
    is a file built from a snapshot of the ledger, so every upload leaves it
    stale and the rows just added are the ones missing from it. On a fresh
    database there is no index at all and `semantic_tool` raises
    FileNotFoundError on the first question that routes to it.

    Failure here is reported, never fatal. The statement is already committed
    and the rows are queryable by SQL — which is most questions — so refusing
    the upload over an index would discard good work to protect a cache.
    """
    from ..agent.nodes.semantic_tool import build_index

    try:
        return {"indexed": build_index()}
    except Exception as exc:
        return {"indexed": 0, "index_error": f"{type(exc).__name__}: {exc}"}


@app.post("/ask", response_model=Answer)
def ask(payload: Question) -> Answer:
    """Run the graph and return the answer with its verification verdict.

    `verified` is returned alongside every answer rather than filtered on: §6.7
    treats an unverified answer as something to surface honestly, not to hide.
    """
    _require_ledger()

    from ..agent.graph import ask as run_agent

    from ..llm import StructuredOutputError

    try:
        result = run_agent(payload.question)
    except Exception as exc:
        # The SQL tool absorbs throttling behind its own retry budget, but a 429
        # in the planner has nowhere to go and took the whole graph down as an
        # unhandled 500 with a stack trace. Daily-quota exhaustion is not worth
        # retrying — the useful thing is to say so.
        if _is_rate_limit(exc):
            raise HTTPException(503, f"Upstream model is rate limited: {exc}") from exc
        # Retries are already exhausted by the time this arrives. The raw
        # exception embeds the whole JSON Schema, so passing it through put a
        # wall of `{"properties": {...}}` in front of the user where a sentence
        # belonged. What they can act on is: ask again, or rephrase.
        if isinstance(exc, StructuredOutputError):
            raise HTTPException(
                502, "The model did not return a usable plan for that question. "
                     "Try asking it a different way."
            ) from exc
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    return Answer(**result)


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "rate limit" in text.lower()


@app.get("/digest/{period}")
def get_digest(period: str, narrate: bool = True) -> dict:
    """Run the §7 detectors for a YYYY-MM month and return the findings.

    Detectors are idempotent (UNIQUE(kind, subject_id, period)), so refreshing
    this endpoint re-checks the month without duplicating what it already found.
    """
    _require_ledger()
    from ..proactive.checks import run_all
    from ..proactive.digest import digest

    conn = connect()
    try:
        counts = run_all(conn, period)
        return {**digest(conn, period, narrate_it=narrate), "new": counts}
    finally:
        conn.close()


# --- §8 human-in-the-loop ----------------------------------------------------

@app.post("/approvals")
def create_approval(payload: ProposalRequest) -> dict:
    """Propose a change. Returns the pending diff and writes nothing."""
    _require_ledger()
    try:
        return propose(payload.action, **payload.params)
    except ApprovalError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/approvals/{thread_id}")
def get_approval(thread_id: str) -> dict:
    proposal = pending(thread_id)
    if proposal is None:
        raise HTTPException(404, f"no approval pending on {thread_id}")
    return {"thread_id": thread_id, "proposal": proposal, "status": "pending"}


@app.post("/approvals/{thread_id}/decide")
def decide_approval(thread_id: str, payload: Decision) -> dict:
    if pending(thread_id) is None:
        raise HTTPException(404, f"no approval pending on {thread_id}")
    try:
        return decide(thread_id, approved=payload.approved, note=payload.note)
    except StaleProposal as exc:
        # 409, not 422: the request was valid when it was made and the ledger
        # moved. The fix is to re-propose and look at the current diff.
        raise HTTPException(409, str(exc)) from exc
    except ApprovalError as exc:
        raise HTTPException(422, str(exc)) from exc


def _require_ledger() -> None:
    if not DB_PATH.exists():
        raise HTTPException(409, "No ledger yet — upload a statement via POST /ingest first")
