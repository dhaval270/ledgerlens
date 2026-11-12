"""Content hashing for idempotent ingestion — §5.2.

Overlapping statement date ranges cost nothing: rows are inserted with
INSERT OR IGNORE against the UNIQUE content_hash column.
"""

from __future__ import annotations

import hashlib


def content_hash(account_id: int, posted_date: str, amount: float, raw_descriptor: str) -> str:
    payload = f"{account_id}|{posted_date}|{amount}|{raw_descriptor}"
    return hashlib.sha256(payload.encode()).hexdigest()
