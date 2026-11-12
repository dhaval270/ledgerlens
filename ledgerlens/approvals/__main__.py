"""Approval CLI — the §8 demo surface.

    python -m ledgerlens.approvals recategorize merchant_id=42 category=health
    python -m ledgerlens.approvals set_budget category=dining limit_amount=300
    python -m ledgerlens.approvals dismiss_insight insight_id=7
    python -m ledgerlens.approvals merge_merchants source_id=9 target_id=4

Prints the pending diff, waits for y/n, then reports what changed. `--yes` skips
the prompt for scripted runs; the graph still pauses and is still resumed with
an explicit decision, because the interrupt is the mechanism and bypassing it
would make the flag a second, unaudited way to write.
"""

from __future__ import annotations

import argparse
import json
import sys

from .actions import ACTIONS, ApprovalError
from .graph import decide, propose

# Parameter names are typed on the command line, so they arrive as strings.
NUMERIC = {"merchant_id", "source_id", "target_id", "insight_id", "limit_amount"}


def _parse_params(pairs: list[str]) -> dict:
    params: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"expected key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        params[key] = float(value) if key == "limit_amount" else (
            int(value) if key in NUMERIC else value
        )
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ledgerlens.approvals")
    parser.add_argument("action", choices=sorted(ACTIONS))
    parser.add_argument("params", nargs="*", metavar="key=value")
    parser.add_argument("--yes", action="store_true", help="approve without prompting")
    args = parser.parse_args(argv)

    try:
        handle = propose(args.action, **_parse_params(args.params))
    except ApprovalError as exc:
        print(f"cannot propose: {exc}", file=sys.stderr)
        return 2

    proposal = handle["proposal"]
    print(f"\nPENDING  {proposal['summary']}")
    print(f"  before {json.dumps(proposal['before'], default=str)}")
    print(f"  after  {json.dumps(proposal['after'], default=str)}")
    print(f"  ({proposal['affected_rows']} transactions affected, "
          f"nothing written yet)\n")

    # No answer is not an answer: EOF or Ctrl-C means the change does not happen.
    try:
        approved = args.yes or input("apply? [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        print("\nno decision — nothing changed.")
        approved = False
    try:
        outcome = decide(handle["thread_id"], approved=approved)
    except ApprovalError as exc:
        print(f"not applied: {exc}", file=sys.stderr)
        return 3

    if outcome["status"] == "rejected":
        print("rejected — nothing changed.")
        return 0
    print(f"applied  {json.dumps(outcome['result'], default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
