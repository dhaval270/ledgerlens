"""Eval harness entry point — §9. Runs the suites behind the README table.

    python evals/run_evals.py              # verifier + golden set
    python evals/run_evals.py --routing    # ...and the full-graph routing suite
    python evals/run_evals.py --rebuild    # ...and merchant/categorization
    python evals/run_evals.py --verifier   # deterministic only, no API calls

The merchant and categorization suites are opt-in because they re-ingest the
synthetic ledger from scratch, which deletes ledger.db. That is correct for a
benchmark and destructive for anyone with real statements loaded, so it does not
happen unless asked for.

Routing is opt-in for a different reason: cost. It runs each question through
the whole graph rather than calling the SQL tool directly, at roughly 3-5 model
calls per question against 1 — about 57k tokens for its 18 queries, against a
free-tier budget of 200k a day. Running it and the golden set back to back is
most of a day's quota, and a suite that exhausts the budget partway through
reports the shortfall as a quality regression.

Suites live in their own modules; this file only sequences them and prints the
table. Each remains runnable alone — a benchmark you can only run all of is a
benchmark you stop running.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import _benchmark  # noqa: F401  (pins LEDGERLENS_DB before ledgerlens loads)

import run_golden_eval  # noqa: E402
import run_merchant_eval  # noqa: E402
import run_routing_eval  # noqa: E402
import run_verifier_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(prog="python evals/run_evals.py")
    ap.add_argument("--rebuild", action="store_true",
                    help="also run merchant/categorization — DELETES and rebuilds ledger.db")
    ap.add_argument("--routing", action="store_true",
                    help="also run the full-graph routing suite (~57k tokens)")
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

    if args.routing:
        print("\n=== routing, through the whole graph (§6.2) ===")
        routing = run_routing_eval.run(run_routing_eval.load("routing"))
        if not routing["n"]:
            print("  every query was rate limited — nothing measured")
        else:
            if routing["rate_limited"]:
                print(f"  !! INCOMPLETE — {routing['rate_limited']} of "
                      f"{routing['attempted']} queries never reached the model")
            print(f"  routing accuracy                 "
                  f"{routing['routing_accuracy_pct']:.1f}%  (n={routing['n']})")
            print(f"  answer accuracy                  "
                  f"{routing['answer_accuracy_pct']:.1f}%")
            print(f"  verified AND correct             {routing['verified_and_correct']}")
            print(f"  verified BUT WRONG               {routing['verified_but_wrong']}")

    if not args.rebuild:
        print("\n  (merchant + categorization skipped; pass --rebuild to run them)")
        if not args.routing:
            print("  (routing skipped; pass --routing to run it)")
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
