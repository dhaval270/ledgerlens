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
    """The full DDL, comments and indexes included."""
    return SCHEMA_PATH.read_text()


def compact_schema(db_path: Path | str = DB_PATH, include_empty: bool = False) -> str:
    """Column list per populated table, for prompts.

    The full DDL was 67% of every SQL-generation prompt — ~4.3KB of comments,
    CHECK constraints, indexes and empty tables resent on every question. On a
    100k tokens/day quota that is roughly one benchmark run per day, and none of
    it helps the model pick columns.

    Reads live tables rather than parsing schema.sql, so empty tables can be
    dropped: a table with no rows can only ever return nothing, and offering it
    invites queries against it (the model answered a subscription question from
    the empty `insights` table).
    """
    conn = connect_readonly(db_path)
    try:
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        lines = []
        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if not count and not include_empty:
                continue
            cols = [
                f"{r[1]} {r[2]}"
                for r in conn.execute(f"PRAGMA table_info({table})")
            ]
            lines.append(f"{table}({', '.join(cols)})  -- {count} rows")
        return "\n".join(lines)
    finally:
        conn.close()
