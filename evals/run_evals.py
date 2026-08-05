"""Eval harness entry point — §9. Runs the suites behind the README table.

    python evals/run_evals.py              # verifier + golden set
    python evals/run_evals.py --rebuild    # ...and merchant/categorization
    python evals/run_evals.py --verifier   # deterministic only, no API calls

The merchant and categorization suites are opt-in because they re-ingest the
synthetic ledger from scratch, which deletes ledger.db. That is correct for a
benchmark and destructive for anyone with real statements loaded, so it does not
happen unless asked for.

Suites live in their own modules; this file only sequences them and prints the
table. Each remains runnable alone — a benchmark you can only run all of is a
benchmark you stop running.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_golden_eval  # noqa: E402
import run_merchant_eval  # noqa: E402
import run_verifier_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(prog="python evals/run_evals.py")
    ap.add_argument("--rebuild", action="store_true",
                    help="also run merchant/categorization — DELETES and rebuilds ledger.db")
    ap.add_argument("--verifier", action="store_true",
                    help="deterministic suite only; makes no API calls")
    args = ap.parse_args()

    print("=== verifier (§9.3) ===")
    run_verifier_eval.main()

    if args.verifier:
        return 0

    print("\n=== golden set (§9.1) ===")
    report = run_golden_eval.run()
    report["refusals"] = run_golden_eval.run_refusals()

    print("\n=== §9.3 table ===")
    print(f"  execution accuracy, answerable   {report['execution_accuracy_pct']:.1f}%  "
          f"(n={report['n']})")
    print(f"  refusal accuracy, unanswerable   "
          f"{report['refusals']['refusal_accuracy_pct']:.1f}%  (n={report['refusals']['n']})")
    print(f"  SQL valid first / after repair   "
          f"{report['sql_valid_first_attempt_pct']:.1f}% / {report['sql_valid_final_pct']:.1f}%")
    print(f"  median / p90 latency             "
          f"{report['median_latency_s']:.2f}s / {report['p90_latency_s']:.2f}s")
    print(f"  median tokens in / out           "
          f"{report['median_tokens_in']:g} / {report['median_tokens_out']:g}")
    print(f"  cost per query                   ${report['usd_per_query']:.5f}")

    if not args.rebuild:
        print("\n  (merchant + categorization skipped; pass --rebuild to run them)")
        return 0

    print("\n=== merchant + categorization (§9.2) — rebuilding ledger.db ===")
    # main() parses sys.argv; call the pieces directly so its flags stay its own.
    stats = run_merchant_eval.build()
    merchants = run_merchant_eval.score()
    categories = run_merchant_eval.score_categories()
    print(f"  merchant cluster accuracy        {merchants['cluster_accuracy_pct']:.1f}%  "
          f"({merchants['true_merchants']} labeled merchants)")
    print(f"  merchant exact match             {merchants['exact_accuracy_pct']:.1f}%")
    print(f"  resolved without an LLM call     {stats['no_llm_pct']:.1f}%")
    print(f"  categorization accuracy          {categories['accuracy_pct']:.1f}%  "
          f"({categories['labels']} labels)")
    print(f"  categorized without an LLM call  {stats['cat_no_llm_pct']:.1f}%")
    print("\n  rules-on accuracy is flattered by construction — the seeded regexes were")
    print("  written knowing this catalog. For the honest number:")
    print("    python evals/run_merchant_eval.py --ablate-rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
