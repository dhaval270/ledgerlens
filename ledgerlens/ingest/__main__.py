"""CLI ingestion — §5. `python -m ledgerlens.ingest <file>...`

Keeps setup reproducible for §12's "under five commands":

    python -m ledgerlens.synthetic
    python -m ledgerlens.ingest --init data/synthetic/transactions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..db import DB_PATH, connect, init_db
from . import ingest_file
from .parse import StatementError


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m ledgerlens.ingest")
    ap.add_argument("files", nargs="+", type=Path, help="statement files (.csv/.pdf)")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--init", action="store_true", help="create the schema first")
    ap.add_argument("--reset", action="store_true", help="delete the db, then init")
    args = ap.parse_args()

    if args.reset:
        args.db.unlink(missing_ok=True)
    if args.init or args.reset or not args.db.exists():
        init_db(args.db)
        print(f"initialized {args.db}")

    failures = 0
    with connect(args.db) as conn:
        for path in args.files:
            try:
                r = ingest_file(conn, path)
            except StatementError as exc:
                print(f"  {path.name}: SKIPPED — {exc}", file=sys.stderr)
                failures += 1
                continue
            print(
                f"  {r.source_file}: {r.parsed} parsed, "
                f"{r.inserted} inserted, {r.duplicates} duplicate"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
