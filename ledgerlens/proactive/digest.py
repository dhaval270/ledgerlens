"""LLM narration of the monthly digest — §7.

The model receives findings from checks.py and **never raw transactions**. It
narrates; it does not compute. Every figure in the digest already exists in an
insight payload, so the same verification argument as §6.7 applies: nothing is
printed that a detector did not produce.

That constraint is also the privacy story. The digest prompt carries aggregates
and merchant names, not a line-by-line statement — so the amount of personal
data leaving the machine is a handful of findings rather than a year of
spending, which matters now that inference is hosted rather than local.
"""

from __future__ import annotations

import sqlite3

from .checks import undismissed

PROMPT = """Write a short monthly summary of these findings about someone's spending.

Period: {period}

Findings (JSON):
{findings}

Rules:
- Two to five sentences. Plain, calm, second person.
- Use ONLY figures that appear in the findings. Do not compute new ones, do not
  add totals, do not estimate.
- Lead with whatever costs the most money or is most actionable.
- State facts, not advice. "Your Spotify charge rose to $13.79" — never
  "you should cancel Spotify".
- If a finding is a drop in spending, do not describe it as a problem."""


def narrate(conn: sqlite3.Connection, period: str) -> str:
    """Turn undismissed insights for a YYYY-MM period into a short digest."""
    findings = undismissed(conn, period)
    if not findings:
        return f"Nothing notable in {period}."

    from ..llm import get_llm

    import json

    response = get_llm().invoke(
        PROMPT.format(period=period, findings=json.dumps(findings, indent=2, default=str))
    )
    return response.content.strip()


def digest(conn: sqlite3.Connection, period: str, narrate_it: bool = True) -> dict:
    """Findings plus optional prose. Findings are the product; prose is a view."""
    findings = undismissed(conn, period)
    return {
        "period": period,
        "count": len(findings),
        "findings": findings,
        "summary": narrate(conn, period) if narrate_it else "",
    }
