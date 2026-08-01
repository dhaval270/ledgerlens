"""Ingestion orchestration — §5. Deterministic, no agent.

    parse → hash → INSERT OR IGNORE → (merchant + category tiers)

Re-uploading a statement is a no-op by construction: §5.2's content_hash is
UNIQUE, so overlapping date ranges cost nothing. That matters more with manual
upload than with a watched folder, because people re-upload constantly.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .dedup import content_hash
from .parse import RawRow, detect_adapter, parse

__all__ = ["IngestResult", "ingest_file", "parse", "RawRow"]


@dataclass
class IngestResult:
    source_file: str
    adapter: str
    parsed: int
    inserted: int
    duplicates: int

    def as_dict(self) -> dict:
        return asdict(self)


def ensure_account(conn: sqlite3.Connection, name: str = "Primary Checking") -> int:
    row = conn.execute("SELECT id FROM accounts WHERE name = ?", (name,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO accounts (name, institution, currency) VALUES (?, ?, ?)",
        (name, "unknown", "USD"),
    )
    return cur.lastrowid


def ingest_file(conn: sqlite3.Connection, path: Path, account_id: int | None = None) -> IngestResult:
    """Parse and load one statement. Raises parse.StatementError on bad input."""
    adapter = detect_adapter(path)
    rows: list[RawRow] = adapter.parse(path)

    if account_id is None:
        account_id = ensure_account(conn)

    now = datetime.now(timezone.utc).isoformat()
    inserted = 0

    for row in rows:
        digest = content_hash(account_id, row.posted_date, row.amount, row.descriptor)
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO transactions
                (account_id, posted_date, amount, currency, raw_descriptor,
                 type, source_file, content_hash, created_at)
            VALUES (?, ?, ?, 'USD', ?, ?, ?, ?, ?)
            """,
            (account_id, row.posted_date, row.amount, row.descriptor,
             row.txn_type, path.name, digest, now),
        )
        inserted += cur.rowcount

    conn.commit()

    return IngestResult(
        source_file=path.name,
        adapter=adapter.name,
        parsed=len(rows),
        inserted=inserted,
        duplicates=len(rows) - inserted,
    )
