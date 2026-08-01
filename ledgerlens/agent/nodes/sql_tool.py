"""SQL tool with self-correcting repair loop — §6.4.

    generate SQL → execute on READ-ONLY connection
      ├─ success   → return rows
      └─ exception → append error to prompt, regenerate (max 3 attempts)
           └─ exhausted → {"error": ..., "rows": []}

Two independent guards, belt and braces:
  1. the connection is opened read-only (db.connect_readonly)
  2. generated SQL containing a mutating keyword is rejected before execution

Always return both the SQL and the rows — the verifier needs both.
"""

from __future__ import annotations

from ..state import AgentState

MAX_SQL_ATTEMPTS = 3

FORBIDDEN = ("DROP", "DELETE", "UPDATE", "INSERT", "ATTACH", "PRAGMA")


def is_safe(sql: str) -> bool:
    """Reject any generated statement that could mutate or reconfigure the db."""
    raise NotImplementedError


def sql_tool(state: AgentState) -> AgentState:
    raise NotImplementedError
