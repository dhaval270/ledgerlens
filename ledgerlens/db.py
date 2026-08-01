"""Connection helpers. §6.4 requires that agent-generated SQL runs read-only."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "ledger.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Read-write connection. Ingestion and approved writes only."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def connect_readonly(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Read-only connection for the SQL tool.

    A malformed or malicious generated query then physically cannot mutate data,
    independent of the keyword blocklist in nodes/sql_tool.py.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DB_PATH) -> None:
    """Apply schema.sql to a fresh database."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text())


def schema_ddl() -> str:
    """The DDL handed to the planner and SQL-generation prompts."""
    return SCHEMA_PATH.read_text()
