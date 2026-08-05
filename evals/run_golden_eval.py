"""Golden-set execution accuracy — §9.3.

Reports:
  execution accuracy      % of questions whose final value matches ground truth
  SQL validity (attempt 1) % that executed without error first try
  SQL validity (final)     % that executed after repair
  repair lift              what the repair loop actually bought
  latency                  median and p90 per question
  tokens                   in/out per question, and cost at the published rate
  refusal accuracy         % of unanswerable questions correctly refused

Comparison is value-based, not SQL-based: any query reaching the right number
counts. Numeric answers use a tolerance because §6.7 allows rounding — a model
returning 53.52333 for an expected 53.5 is right, not wrong.

    python evals/run_golden_eval.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # evals/ is not a package

from langchain_core.callbacks import get_usage_metadata_callback

from ledgerlens.agent.nodes.sql_tool import run_query

GOLDEN = Path(__file__).parent / "golden_queries.jsonl"

ABS_TOLERANCE = 0.02      # cents
REL_TOLERANCE = 0.005     # 0.5%, covers rounding of percentages

# Token counts below are measured; the dollar figure is derived from Groq's
# published rate for llama-3.3-70b-versatile at the time of writing. Kept as a
# constant so the cost line can be corrected without re-running the benchmark.
USD_PER_MTOK_IN = 0.59
USD_PER_MTOK_OUT = 0.79


def _scalar(rows: list[dict]):
    """The value a scalar question should have produced.

    A single-column single row is unambiguous. Multiple columns are returned
    whole and unpacked by `matches`, because a model answering "what was my
    largest purchase?" with {canonical_name: 'Greenline', amount: -1450.0} has
    answered correctly — insisting on one column would score presentation, not
    accuracy.
    """
    if not rows:
        return None
    first = rows[0]
    if len(first) == 1:
        return next(iter(first.values()))
    return first


def matches(expected, got) -> bool:
    if got is None:
        return False

    # A multi-column row counts if any cell carries the expected answer.
    if isinstance(got, dict):
        return any(matches(expected, value) for value in got.values())

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            got_num = float(got)
        except (TypeError, ValueError):
            return False
        diff = abs(got_num - float(expected))
        return diff <= max(ABS_TOLERANCE, abs(float(expected)) * REL_TOLERANCE)

    if isinstance(expected, str):
        a, b = str(expected).strip().lower(), str(got).strip().lower()
        return a == b or a in b or b in a

    return expected == got


def run(limit: int | None = None) -> dict:
    entries = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    answerable = [e for e in entries if not e["expect_refusal"]]
    if limit:
        answerable = answerable[:limit]

    results = []
    for entry in answerable:
        started = time.time()
        # Tokens are read off the callback rather than estimated: the prompt
        # carries a live schema and vocabulary, so its size is not knowable
        # from the source.
        with get_usage_metadata_callback() as usage:
            outcome = run_query(entry["question"])
        elapsed = time.time() - started

        got = _scalar(outcome["rows"])
        ok = matches(entry["expected_value"], got)

        results.append({
            "id": entry["id"],
            "question": entry["question"],
            "expected": entry["expected_value"],
            "got": got,
            "correct": ok,
            "attempts": outcome["attempts"],
            "executed": outcome["error"] is None,
            "error": outcome["error"],
            "sql": outcome["query"],
            "seconds": elapsed,
            **_tokens(usage.usage_metadata),
        })
        print(("  PASS " if ok else "  FAIL ") + f"{entry['id']:11} "
              f"expected={entry['expected_value']!r} got={got!r} "
              f"attempts={outcome['attempts']}", flush=True)

    n = len(results)
    executed = [r for r in results if r["executed"]]
    first_try = [r for r in results if r["executed"] and r["attempts"] == 1]
    latencies = [r["seconds"] for r in results]
    tokens_in = [r["tokens_in"] for r in results]
    tokens_out = [r["tokens_out"] for r in results]

    return {
        "n": n,
        "execution_accuracy_pct": sum(r["correct"] for r in results) / n * 100,
        "sql_valid_first_attempt_pct": len(first_try) / n * 100,
        "sql_valid_final_pct": len(executed) / n * 100,
        "repaired": len(executed) - len(first_try),
        "median_latency_s": st.median(latencies),
        "p90_latency_s": sorted(latencies)[int(n * 0.9)] if n > 1 else latencies[0],
        "median_tokens_in": st.median(tokens_in),
        "median_tokens_out": st.median(tokens_out),
        "total_tokens": sum(tokens_in) + sum(tokens_out),
        "usd_per_query": (
            st.mean(tokens_in) / 1e6 * USD_PER_MTOK_IN
            + st.mean(tokens_out) / 1e6 * USD_PER_MTOK_OUT
        ),
        "results": results,
    }


def _tokens(usage: dict) -> dict:
    """Sum across models — one question can touch more than one."""
    return {
        "tokens_in": sum(v.get("input_tokens", 0) for v in usage.values()),
        "tokens_out": sum(v.get("output_tokens", 0) for v in usage.values()),
    }


def run_refusals() -> dict:
    """§9.1 — the deliberately unanswerable questions.

    Scored through the planner rather than the SQL tool, because refusing is a
    planning decision: the failure mode being tested is a model that cheerfully
    writes SQL for "what is my credit score?" and returns a number the ledger
    cannot support. A refusal here is the correct answer, not an abstention.
    """
    from ledgerlens.agent.nodes.planner import make_plan

    entries = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    refusals = [e for e in entries if e["expect_refusal"]]

    results = []
    for entry in refusals:
        with get_usage_metadata_callback() as usage:
            plan = make_plan(entry["question"])
        refused = not plan.answerable
        results.append({
            "id": entry["id"],
            "question": entry["question"],
            "refused": refused,
            "reason": plan.refusal_reason,
            **_tokens(usage.usage_metadata),
        })
        print(("  PASS " if refused else "  FAIL ") +
              f"{entry['id']:11} {entry['question'][:46]:46} "
              f"refused={refused}", flush=True)

    n = len(results)
    return {
        "n": n,
        "refusal_accuracy_pct": sum(r["refused"] for r in results) / n * 100,
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-refusals", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "golden_results.json")
    args = ap.parse_args()

    report = run(args.limit)
    if not args.skip_refusals:
        print("\n--- refusals ---")
        report["refusals"] = run_refusals()
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print("\n=== §9.3 metrics ===")
    print(f"  execution accuracy        {report['execution_accuracy_pct']:.1f}%  (n={report['n']})")
    print(f"  SQL valid, first attempt  {report['sql_valid_first_attempt_pct']:.1f}%")
    print(f"  SQL valid, after repair   {report['sql_valid_final_pct']:.1f}%")
    print(f"  queries fixed by repair   {report['repaired']}")
    print(f"  median latency            {report['median_latency_s']:.2f}s")
    print(f"  p90 latency               {report['p90_latency_s']:.2f}s")
    print(f"  median tokens in/out      {report['median_tokens_in']:g} / "
          f"{report['median_tokens_out']:g}")
    print(f"  cost per query            ${report['usd_per_query']:.5f}  "
          f"({report['total_tokens']:,} tokens this run)")
    if "refusals" in report:
        print(f"  refusal accuracy          {report['refusals']['refusal_accuracy_pct']:.1f}%  "
              f"(n={report['refusals']['n']})")

    failures = [r for r in report["results"] if not r["correct"]]
    if failures:
        print(f"\n=== {len(failures)} failures ===")
        for r in failures:
            print(f"\n  [{r['id']}] {r['question']}")
            print(f"    expected {r['expected']!r}, got {r['got']!r}")
            if r["error"]:
                print(f"    error: {r['error']}")
            print(f"    sql: {r['sql'][:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
