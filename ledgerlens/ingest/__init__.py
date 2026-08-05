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

__all__ = ["IngestResult", "ingest_file", "parse", "RawRow", "resolve_pending"]


@dataclass
class IngestResult:
    source_file: str
    adapter: str
    parsed: int
    inserted: int
    duplicates: int

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_pending(conn: sqlite3.Connection, use_llm: bool = True) -> dict:
    """Resolve merchants and categories for rows that have none, then detect series.

    Lives here rather than in the CLI because upload is now the primary path and
    was skipping it entirely: rows ingested through the API arrived with NULL
    merchant and category, which the UI renders as an em dash and — worse — makes
    invisible to any query that joins `merchants`. A row you can see in a table
    but cannot reach with a join is a row that silently drops out of answers.

    Ordering is not optional: recurring detection groups by merchant_id, so it
    has nothing to group on until resolution has run.
    """
    from ..seed import seed
    from .categorize import categorize
    from .merchants import resolve
    from .recurring import detect_series, price_hikes

    seed(conn)
    rows = conn.execute(
        "SELECT id, raw_descriptor, type FROM transactions WHERE merchant_id IS NULL"
    ).fetchall()

    llm_calls = 0
    for txn_id, descriptor, txn_type in rows:
        merchant = resolve(conn, descriptor, use_llm=use_llm)
        category = categorize(conn, descriptor, merchant.merchant_id, txn_type,
                              use_llm=use_llm)
        llm_calls += (merchant.tier == 3) + (category.tier == 4)
        conn.execute(
            """UPDATE transactions
               SET merchant_id = ?, category_id = ?, categorized_by = ?, confidence = ?
               WHERE id = ?""",
            (merchant.merchant_id, category.category_id,
             category.categorized_by, category.confidence, txn_id),
        )
    conn.commit()

    return {
        "resolved": len(rows),
        "llm_calls": llm_calls,
        "recurring_series": detect_series(conn),   # returns a count, not a list
        "price_hikes": len(price_hikes(conn)),
    }


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
