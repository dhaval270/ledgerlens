"""End-to-end routing accuracy — the metric §9.3 never actually collected.

`run_golden_eval.py` calls `run_query()` directly. That measures the SQL tool,
which is a real and useful number, but it means the planner never ran, the
graph was never built, and `semantic_tool`/`anomaly_tool` never executed under
evaluation. The `tools` field in the golden set was written, committed, tested
for well-formedness — and read by nothing. Two tools sat in the architecture
diagram, in the README, and in the routing table with zero measured behaviour
behind them.

This harness runs `agent.graph.ask()`, so a question passes through planner →
tool(s) → verifier → answer exactly as it does from the API, and scores three
things the SQL-only path cannot see:

  routing accuracy    did the planner select the tools the question needs
  answer accuracy     did the expected value survive to a retrieved row
  verification rate   did the verifier pass what turned out to be correct

The third is the interesting one. Execution accuracy and verification are
independent — an answer can be right and unverified, or verified and wrong —
and the cross-tab of the two is the only place the verifier's real precision
becomes visible.

    python evals/run_routing_eval.py                 # routing subset (18 queries)
    python evals/run_routing_eval.py --subset all    # every answerable query
    python evals/run_routing_eval.py --subset refusals

Costs roughly 3-5 LLM calls per question, against 1 for the SQL-only harness.
The subset default is deliberate: the routing queries are the ones with no
existing measurement, and running all 59 is several times the token spend.
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

from evals.run_golden_eval import USD_PER_MTOK_IN, USD_PER_MTOK_OUT, _tokens, matches

GOLDEN = Path(__file__).parent / "golden_queries.jsonl"


def load(subset: str) -> list[dict]:
    entries = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    if subset == "refusals":
        return [e for e in entries if e["expect_refusal"]]
    answerable = [e for e in entries if not e["expect_refusal"]]
    if subset == "all":
        return answerable
    return [e for e in answerable if e["tools"] != ["sql"]]


def tools_used(outcome: dict) -> set[str]:
    """What actually ran, read off tool results rather than off the plan.

    The plan states intent; `tool_results` states fact. They diverge — the graph
    falls back to running a tool over the raw question when the plan named no
    step for it, so a plan can promise `semantic` and a run can deliver `sql`.
    Scoring the plan would credit the intention.
    """
    return {r.get("tool") for r in outcome.get("tool_results", []) if r.get("tool")}


def value_found(outcome: dict, expected) -> bool:
    """Did the expected value reach a retrieved row?

    Checked against rows, not against the answer prose. §6.2's answer node is
    string assembly over the same rows, so a value present in the rows and
    absent from the answer is a formatting bug, not a retrieval one — and
    scoring prose would conflate the two.
    """
    for result in outcome.get("tool_results", []):
        for row in result.get("rows") or []:
            for cell in row.values():
                if matches(expected, cell):
                    return True
    return False


def run(entries: list[dict], refusals: bool = False) -> dict:
    from ledgerlens.agent.graph import ask

    results = []
    for entry in entries:
        started = time.time()
        with get_usage_metadata_callback() as usage:
            try:
                outcome = ask(entry["question"])
                failed = None
            except Exception as exc:                     # a crash is a result
                outcome, failed = {}, f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - started

        if refusals:
            ok = failed is None and not outcome.get("answerable", True)
            record = {"id": entry["id"], "question": entry["question"],
                      "refused": ok, "correct": ok, "error": failed,
                      "reason": (outcome.get("verdict") or {}).get("reason", "")}
            print(("  PASS " if ok else "  FAIL ") +
                  f"{entry['id']:9} refused={ok}", flush=True)
        else:
            used = tools_used(outcome)
            expected = set(entry["tools"])
            record = {
                "id": entry["id"],
                "question": entry["question"],
                "expected_tools": sorted(expected),
                "used_tools": sorted(used),
                # Superset, not equality: reaching for SQL alongside semantic is
                # the documented pattern (§6.5), and an extra tool that does not
                # change the answer is a cost problem, not a routing error.
                "routed": bool(expected) and expected <= used,
                "correct": value_found(outcome, entry["expected_value"]),
                "verified": bool(outcome.get("verified")),
                "no_data": bool(outcome.get("no_data")),
                "error": failed,
            }
            print(f"  {'PASS' if record['correct'] else 'FAIL'} "
                  f"{'ROUTE' if record['routed'] else '  X  '} "
                  f"{entry['id']:9} want={sorted(expected)} got={sorted(used)}",
                  flush=True)

        record["seconds"] = elapsed
        record.update(_tokens(usage.usage_metadata))
        results.append(record)

    n = len(results)
    report = {
        "n": n,
        "answer_accuracy_pct": sum(r["correct"] for r in results) / n * 100,
        "median_latency_s": st.median(r["seconds"] for r in results),
        "total_tokens": sum(r["tokens_in"] + r["tokens_out"] for r in results),
        "usd_per_query": (
            st.mean([r["tokens_in"] for r in results]) / 1e6 * USD_PER_MTOK_IN
            + st.mean([r["tokens_out"] for r in results]) / 1e6 * USD_PER_MTOK_OUT
        ),
        "results": results,
    }

    if not refusals:
        report["routing_accuracy_pct"] = sum(r["routed"] for r in results) / n * 100
        report["per_tool"] = _per_tool(results)
        # The cross-tab: verification is only meaningful next to correctness.
        report["verified_and_correct"] = sum(r["verified"] and r["correct"] for r in results)
        report["verified_but_wrong"] = sum(r["verified"] and not r["correct"] for r in results)
        report["correct_but_unverified"] = sum(r["correct"] and not r["verified"] for r in results)

    return report


def _per_tool(results: list[dict]) -> dict:
    per: dict[str, dict] = {}
    for r in results:
        for tool in r["expected_tools"]:
            bucket = per.setdefault(tool, {"n": 0, "routed": 0, "correct": 0})
            bucket["n"] += 1
            bucket["routed"] += r["routed"]
            bucket["correct"] += r["correct"]
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["routing", "all", "refusals"], default="routing")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "routing_results.json")
    args = ap.parse_args()

    entries = load(args.subset)
    if args.limit:
        entries = entries[:args.limit]
    if not entries:
        print("no queries matched that subset", file=sys.stderr)
        return 1

    print(f"=== {args.subset}: {len(entries)} queries through the full graph ===")
    report = run(entries, refusals=args.subset == "refusals")
    report["subset"] = args.subset
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print("\n=== routing metrics ===")
    print(f"  answer accuracy           {report['answer_accuracy_pct']:.1f}%  (n={report['n']})")
    if "routing_accuracy_pct" in report:
        print(f"  routing accuracy          {report['routing_accuracy_pct']:.1f}%")
        for tool, b in sorted(report["per_tool"].items()):
            print(f"    {tool:9} n={b['n']:3}  routed {b['routed']:3}  correct {b['correct']:3}")
        print(f"  verified AND correct      {report['verified_and_correct']}")
        print(f"  verified BUT WRONG        {report['verified_but_wrong']}")
        print(f"  correct but unverified    {report['correct_but_unverified']}")
    print(f"  median latency            {report['median_latency_s']:.2f}s")
    print(f"  cost per query            ${report['usd_per_query']:.5f}  "
          f"({report['total_tokens']:,} tokens)")

    failures = [r for r in report["results"] if not r["correct"]]
    if failures:
        print(f"\n=== {len(failures)} failures ===")
        for r in failures:
            print(f"  [{r['id']}] {r['question']}")
            if r.get("error"):
                print(f"    crashed: {r['error']}")
            elif "used_tools" in r:
                print(f"    wanted {r['expected_tools']}, ran {r['used_tools']}, "
                      f"verified={r['verified']} no_data={r['no_data']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
