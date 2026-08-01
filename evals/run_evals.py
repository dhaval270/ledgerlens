"""Eval harness — §9. Non-optional; this is what separates the project from a tutorial.

Metrics reported into the README table (§9.3):
  - execution accuracy on the golden set (% correct final value)
  - SQL validity rate, first attempt vs. after repair
  - verifier catch rate — inject wrong numbers, confirm rejection
  - categorization accuracy, and % resolved without an LLM call
  - median latency and token cost per query
"""

from __future__ import annotations

from pathlib import Path

GOLDEN = Path(__file__).parent / "golden_queries.jsonl"
LABELED = Path(__file__).parent / "labeled_categories.csv"


def run_golden_queries() -> dict:
    """Execution accuracy + SQL validity across golden_queries.jsonl."""
    raise NotImplementedError


def run_verifier_catch_rate() -> dict:
    """Inject wrong numbers into known-good answers; confirm the verifier rejects."""
    raise NotImplementedError


def run_categorization_accuracy() -> dict:
    """Accuracy overall and per resolution tier against labeled_categories.csv."""
    raise NotImplementedError


def main() -> int:
    """Run every suite and print the §9.3 metrics table."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
