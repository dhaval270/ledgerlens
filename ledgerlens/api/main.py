"""FastAPI surface — §3.

Statements are uploaded manually, so /ingest is the primary entry point rather
than §5's watched folder. Uploads land in data/private/, which is gitignored
per §10.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..db import DB_PATH, connect, init_db
from ..ingest import ingest_file
from ..ingest.parse import (
    EncryptedStatement,
    NoTextLayer,
    StatementError,
    UnknownFormat,
)

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "private"
ALLOWED_SUFFIXES = {".pdf", ".csv"}

app = FastAPI(title="LedgerLens", version="0.1.0")


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    verified: bool
    plan: list[dict]
    tool_results: list[dict]


@app.get("/")
def root() -> dict:
    """Landing route — hitting / and getting a 404 reads as a broken server."""
    return {
        "service": "LedgerLens",
        "docs": "/docs",
        "endpoints": {
            "GET /health": "service and database status",
            "POST /ingest": "upload a statement (.csv/.pdf)",
            "POST /ask": "ask a question (not implemented yet)",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "db": DB_PATH.exists()}


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
    except (EncryptedStatement, NoTextLayer, UnknownFormat) as exc:
        # Keep these distinct from a 500 — they are user-fixable input problems,
        # and the message says which one it is.
        raise HTTPException(422, str(exc)) from exc
    except StatementError as exc:
        raise HTTPException(400, str(exc)) from exc

    return result.as_dict()


@app.post("/ask", response_model=Answer)
def ask(payload: Question) -> Answer:
    """Run the graph and return the answer with its verification verdict."""
    raise NotImplementedError
