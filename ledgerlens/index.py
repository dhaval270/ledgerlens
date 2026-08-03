"""Build the semantic index — §6.5. `python -m ledgerlens.index`

Separate from ingestion on purpose: embedding every transaction costs seconds
and loads a ~90MB model, while the deterministic ingest path must stay fast and
offline-capable. Rebuild after ingesting new statements.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .agent.nodes.semantic_tool import INDEX_PATH, build_index
from .db import DB_PATH


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m ledgerlens.index")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--out", type=Path, default=INDEX_PATH)
    args = ap.parse_args()

    if not args.db.exists():
        print(f"no database at {args.db}")
        return 1

    started = time.time()
    count = build_index(args.db, args.out)
    print(f"indexed {count} transactions -> {args.out} ({time.time() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
