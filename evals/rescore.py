"""Re-score a saved golden_results.json with the current comparator.

Scoring bugs are cheaper to fix than re-runs: a full pass is ~46 Groq calls at
~13s each. The raw `got` values are stored, so a corrected comparator can be
applied offline without spending the calls again.

    python evals/rescore.py [results.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_golden_eval import matches


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "golden_results.json"
    report = json.loads(path.read_text())

    changed, correct = [], 0
    for r in report["results"]:
        now = matches(r["expected"], r["got"])
        if now != r["correct"]:
            changed.append((r["id"], r["correct"], now))
        r["correct"] = now
        correct += now

    n = report["n"]
    old = report["execution_accuracy_pct"]
    report["execution_accuracy_pct"] = correct / n * 100

    print(f"{path.name}: {old:.1f}% -> {report['execution_accuracy_pct']:.1f}%  ({correct}/{n})")
    if changed:
        print(f"  {len(changed)} verdict(s) changed by the comparator fix:")
        for qid, was, now in changed:
            print(f"    {qid:11} {'FAIL' if not was else 'PASS'} -> {'PASS' if now else 'FAIL'}")

    print("\n  remaining failures:")
    for r in report["results"]:
        if not r["correct"]:
            print(f"    {r['id']:11} exp={str(r['expected'])[:20]:20} got={str(r['got'])[:34]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
