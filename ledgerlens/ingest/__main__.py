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
    ap.add_argument(
        "--resolve",
        action="store_true",
        help="also run merchant + category resolution and recurring detection "
             "(calls the LLM for unseen merchants)",
    )
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

        if args.resolve:
            _resolve_all(conn)

    return 1 if failures else 0


def _resolve_all(conn) -> None:
    """Merchant + category resolution, then recurring detection.

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
    for tid, descriptor, txn_type in rows:
        m = resolve(conn, descriptor)
        c = categorize(conn, descriptor, m.merchant_id, txn_type)
        llm_calls += (m.tier == 3) + (c.tier == 4)
        conn.execute(
            """UPDATE transactions
               SET merchant_id = ?, category_id = ?, categorized_by = ?, confidence = ?
               WHERE id = ?""",
            (m.merchant_id, c.category_id, c.categorized_by, c.confidence, tid),
        )
    conn.commit()

    print(f"  resolved {len(rows)} transactions ({llm_calls} LLM calls)")

    series = detect_series(conn)
    hikes = price_hikes(conn)
    print(f"  detected {series} recurring series, {len(hikes)} price hike(s)")
    for h in hikes:
        print(f"    {h['merchant']}: {h['typical_amount']:.2f} -> "
              f"{h['last_amount']:.2f} ({h['ratio']:.1%} of typical)")


if __name__ == "__main__":
    raise SystemExit(main())
