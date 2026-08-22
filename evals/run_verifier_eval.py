"""Verifier catch rate — §9.3.

The experiment the spec asks for: inject wrong numbers, confirm rejection.

Two rates, and the second matters as much as the first:

  catch rate        % of corrupted answers correctly rejected
  false rejection   % of *correct* answers wrongly rejected

A verifier that rejects everything scores 100% catch and is worthless. One that
fails correct answers is worse than absent, because it teaches the reader to
ignore the verdict. Both numbers go in the README together.

Runs entirely on reference SQL — no LLM, no network, fully deterministic.

    python evals/run_verifier_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import _benchmark  # noqa: F401  (pins LEDGERLENS_DB before ledgerlens loads)

from ledgerlens.agent.nodes.verifier import verify
from ledgerlens.db import DB_PATH, connect_readonly

GOLDEN = Path(__file__).parent / "golden_queries.jsonl"

# Multiplicative corruptions, chosen to sit well outside rounding tolerance.
CORRUPTIONS = [
    ("inflated 20%", lambda v: v * 1.20),
    ("deflated 35%", lambda v: v * 0.65),
    ("order of magnitude", lambda v: v * 10),
    ("doubled", lambda v: v * 2),
]

# A corruption landing inside verification tolerance is not a miss — it is an
# invalid trial, and counting it would understate the catch rate.
MIN_CORRUPTION_DELTA = 1.0


def _draft(value) -> str:
    if isinstance(value, (int, float)):
        return f"You spent ${abs(value):,.2f} in that period."
    return f"The answer is {value}."


def main() -> int:
    if not DB_PATH.exists():
        print("ledger.db not found", file=sys.stderr)
        return 1

    entries = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    numeric = [
        e for e in entries
        if not e["expect_refusal"]
        and isinstance(e["expected_value"], (int, float))
        and not isinstance(e["expected_value"], bool)
        and abs(e["expected_value"]) >= 10
    ]

    caught = attempted = 0
    false_rejections = []

    from build_golden_queries import DETECTORS

    with connect_readonly() as conn:
        for entry in numeric:
            # Anomaly questions carry `reference_fn` rather than `reference_sql`:
            # SQLite has no STDDEV, so their answer key is the detector itself.
            # This loop assumed every entry had SQL and started raising KeyError
            # the day those questions were added — the suite behind the README's
            # verifier row has not been runnable since.
            if sql := entry.get("reference_sql"):
                rows = [dict(r) for r in conn.execute(sql).fetchall()]
                query = sql
            else:
                value = DETECTORS[entry["reference_fn"]](conn)
                rows = [{"value": value}]
                query = f"{entry['reference_fn']}()  # detector, not SQL"

            tool_results = [{"tool": "sql", "query": query,
                             "rows": rows, "error": None}]

            # 1. the honest answer must pass
            truth = verify(_draft(entry["expected_value"]), tool_results)
            if not truth["pass"]:
                false_rejections.append((entry["id"], truth["reason"]))

            # 2. every corruption must be rejected
            for label, corrupt in CORRUPTIONS:
                bad = corrupt(entry["expected_value"])
                if abs(abs(bad) - abs(entry["expected_value"])) < MIN_CORRUPTION_DELTA:
                    continue  # too close to the truth to be a fair trial
                if abs(bad) < 10:
                    continue  # below MIN_CHECKED_MAGNITUDE, a known blind spot
                attempted += 1
                if not verify(_draft(bad), tool_results)["pass"]:
                    caught += 1

    print(f"=== §9.3 verifier catch rate ===")
    print(f"  numeric questions tested   {len(numeric)}")
    print(f"  corrupted answers injected {attempted}")
    print(f"  caught                     {caught}  ({caught / attempted * 100:.1f}%)")
    print(f"  false rejections           {len(false_rejections)}/{len(numeric)} "
          f"({len(false_rejections) / len(numeric) * 100:.1f}%)")
    if false_rejections:
        print("\n  correct answers wrongly rejected:")
        for qid, reason in false_rejections:
            print(f"    {qid:11} {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
