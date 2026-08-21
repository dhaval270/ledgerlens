"""SQL tool with self-correcting repair loop — §6.4.

    generate SQL → execute on READ-ONLY connection
      ├─ success   → return rows
      └─ exception → append error to prompt, regenerate (max 3 attempts)
           └─ exhausted → {"error": ..., "rows": []}

Two independent guards, belt and braces:

  1. the connection is opened read-only (db.connect_readonly), so a mutating
     statement physically cannot succeed whatever the model emits;
  2. generated SQL is screened before execution.

Guard 2 exists for message quality, not safety — guard 1 is the real one. If the
screen ever disagrees with the connection, the connection wins.

Always returns both the SQL and the rows: the verifier (§6.7) needs the query to
explain a verdict and the rows to check numbers against.
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any

from pydantic import BaseModel, Field

from ...db import compact_schema, connect_readonly
from ..state import AgentState

MAX_SQL_ATTEMPTS = 3
MAX_ROWS = 200

# Throttling is infrastructure, not a query defect, so it gets its own budget.
MAX_THROTTLE_RETRIES = 4
THROTTLE_BACKOFF_S = 4.0

FORBIDDEN = (
    "DROP", "DELETE", "UPDATE", "INSERT", "ATTACH", "PRAGMA",
    "CREATE", "ALTER", "REPLACE", "VACUUM", "DETACH",
)

_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


class GeneratedSQL(BaseModel):
    """Schema-enforced so the model cannot return a markdown-fenced blob."""

    sql: str = Field(description="A single SELECT statement, no trailing semicolon")
    rationale: str = Field(description="One line: what this query computes")


def _scrub(sql: str) -> str:
    """Remove comments and string literals before keyword screening.

    Without this, `WHERE canonical_name = 'Bed Bath & Update'` trips the UPDATE
    check and a legitimate query gets rejected.
    """
    sql = _BLOCK_COMMENT.sub(" ", sql)
    sql = _LINE_COMMENT.sub(" ", sql)
    return _STRING_LITERAL.sub("''", sql)


def is_safe(sql: str) -> tuple[bool, str]:
    """Screen generated SQL. Returns (ok, reason)."""
    scrubbed = _scrub(sql).strip().rstrip(";").strip()
    if not scrubbed:
        return False, "empty query"

    # Multiple statements: a second one is never needed and always suspicious.
    if ";" in scrubbed:
        return False, "multiple statements are not allowed"

    upper = scrubbed.upper()
    if not re.match(r"^\s*(SELECT|WITH)\b", upper):
        return False, "query must begin with SELECT or WITH"

    tokens = set(re.findall(r"\b[A-Z_]+\b", upper))
    hits = sorted(tokens & set(FORBIDDEN))
    if hits:
        return False, f"forbidden keyword(s): {', '.join(hits)}"

    return True, ""


PROMPT = """You write SQLite queries against a personal finance ledger.

Schema:
{schema}

Data covers {date_min} to {date_max}. Today is {date_max}.

Conventions that are always true:
- `amount` is negative for outflow, positive for income and refunds.
- Dates are ISO strings 'YYYY-MM-DD'. Use substr(posted_date,1,7) for a month.
- "How much did I spend" means SUM(amount), which will be negative.
- Join merchants and categories by id; never match on free text unless asked.

Transaction types in this ledger: {types}
{spend_rule}

When the question compares two periods, categories or merchants, GROUP BY that
dimension and return one row per side. A single combined SUM over both does not
answer a comparison — "June vs July" needs two rows, not one total.

Values present in the data:
- categories.name is one of: {categories}
- merchants.canonical_name is one of: {merchants}

Only the tables listed above exist and hold data.

Because outflows are negative, "more" and "largest" invert the usual operator:
- spent MORE  -> more negative -> smaller amount -> MIN, or ORDER BY ... ASC
- spent LESS  -> less negative -> larger amount  -> MAX, or ORDER BY ... DESC
- MAX(amount) over purchases is the *cheapest* one. Never use it for "largest".

When the question asks WHICH month, category or merchant, return that name or
month string — never a raw id.

Question: {question}

Return ONE SELECT statement answering it. No semicolon, no markdown, no prose.
Prefer a single scalar result when the question has a single answer.
{repair}"""

REPAIR = """
Your previous attempt failed.
SQL:   {sql}
Error: {error}
Fix it and return corrected SQL."""


def _type_mix(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "SELECT type, COUNT(*) FROM transactions GROUP BY type ORDER BY 2 DESC"
    ).fetchall()
    return ", ".join(f"{name} ({count})" for name, count in rows) or "none"


def _spend_rule(conn: sqlite3.Connection) -> str:
    """The purchase-filter guidance, decided by the data rather than assumed.

    §4 has spending analytics filter `type = 'purchase'`, which is right for a
    card statement and catastrophic for a checking account: asked "who did I pay
    the most?", the model added `WHERE type = 'purchase'` to a ledger holding
    only transfers and got nothing back. Valid SQL, real schema, empty answer.

    Same reasoning as `compact_schema` dropping empty tables — a filter that can
    only match zero rows should not be suggested, because the model has no way
    to know it is a dead end and every attempt looks reasonable.
    """
    (purchases,) = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE type = 'purchase'"
    ).fetchone()
    if purchases:
        return ("- Spending analytics filter `type = 'purchase'` unless the question is\n"
                "  explicitly about income, transfers, fees or refunds.")
    return ("- There are NO `purchase` rows here, so never filter on that type — it can\n"
            "  only return nothing. Money leaving the account is any negative `amount`,\n"
            "  whatever its type.")


def _date_bounds(conn: sqlite3.Connection) -> tuple[str, str]:
    row = conn.execute("SELECT MIN(posted_date), MAX(posted_date) FROM transactions").fetchone()
    return (row[0] or "unknown", row[1] or "unknown")


def _vocabulary(conn: sqlite3.Connection) -> tuple[str, str]:
    """The literal values the model must filter on.

    The DDL names columns but not their contents, so a question about
    "anything medical-looking" produced `WHERE c.name = 'medical'` — valid SQL
    matching zero rows. Seven of the 22 baseline failures were NULL results of
    exactly this shape. This is missing *context*, not a missing instruction:
    it constrains what the model can write without telling it how to think.
    """
    categories = [r[0] for r in conn.execute("SELECT name FROM categories ORDER BY name")]
    merchants = [
        r[0] for r in conn.execute("SELECT canonical_name FROM merchants ORDER BY canonical_name")
    ]
    return ", ".join(categories), ", ".join(merchants)


def _table_counts(conn: sqlite3.Connection) -> str:
    """Row count per table.

    The schema advertises tables that are empty until later phases run —
    `insights` is populated by §7's detectors, which do not exist yet. Given
    only the DDL, "did any subscription go up in price?" is very reasonably
    answered by querying `insights`, which returns 0 forever. Stating the counts
    lets the model route around tables that cannot answer anything.
    """
    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    lines = []
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        lines.append(f"- {table}: {count}" + ("  (EMPTY — do not use)" if count == 0 else ""))
    return "\n".join(lines)


def generate_sql(question: str, repair: str = "",
                 transaction_ids: list[int] | None = None) -> GeneratedSQL:
    from ...llm import invoke_structured

    with connect_readonly() as conn:
        date_min, date_max = _date_bounds(conn)
        categories, merchants = _vocabulary(conn)
        types, spend_rule = _type_mix(conn), _spend_rule(conn)

    scope = ""
    if transaction_ids:
        ids = ", ".join(str(i) for i in transaction_ids)
        scope = (f"\n\nRestrict the query to these transaction ids, which a "
                 f"semantic search already selected:\n  t.id IN ({ids})")

    return invoke_structured(
        GeneratedSQL,
        scope + PROMPT.format(
            schema=compact_schema(),
            question=question,
            date_min=date_min,
            date_max=date_max,
            categories=categories,
            merchants=merchants,
            types=types,
            spend_rule=spend_rule,
            repair=repair,
        )
    )


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "rate limit" in text.lower()


def run_query(question: str, max_attempts: int = MAX_SQL_ATTEMPTS,
              transaction_ids: list[int] | None = None) -> dict[str, Any]:
    """Generate, screen, execute — repairing on failure. Never raises.

    Transport failures are not SQL failures. A 429 says nothing about the query,
    so feeding it back as repair context asks the model to "fix" working SQL,
    and consuming a repair attempt on it exhausts the budget without ever
    reaching the database. Measured: four golden-set questions returned empty
    SQL for precisely this reason, and it inflated apparent run-to-run variance
    because throttling is intermittent.
    """
    repair = ""
    last_sql = ""
    last_error = ""
    throttle_budget = MAX_THROTTLE_RETRIES

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            generated = generate_sql(question, repair, transaction_ids)
        except Exception as exc:
            if _is_rate_limit(exc) and throttle_budget > 0:
                # Wait it out. Does not consume a repair attempt.
                time.sleep(THROTTLE_BACKOFF_S * (MAX_THROTTLE_RETRIES - throttle_budget + 1))
                throttle_budget -= 1
                attempt -= 1
                continue
            last_error = f"generation failed: {exc}"
            repair = REPAIR.format(sql=last_sql or "(none)", error=last_error)
            continue

        last_sql = generated.sql.strip().rstrip(";")

        ok, reason = is_safe(last_sql)
        if not ok:
            last_error = f"rejected by safety screen: {reason}"
            repair = REPAIR.format(sql=last_sql, error=last_error)
            continue

        try:
            with connect_readonly() as conn:
                cursor = conn.execute(last_sql)
                rows = [dict(r) for r in cursor.fetchmany(MAX_ROWS)]
        except sqlite3.Error as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            repair = REPAIR.format(sql=last_sql, error=last_error)
            continue

        return {
            "tool": "sql",
            "query": last_sql,
            "rationale": generated.rationale,
            "rows": rows,
            "error": None,
            "attempts": attempt,
        }

    return {
        "tool": "sql",
        "query": last_sql,
        "rationale": "",
        "rows": [],
        "error": last_error,
        "attempts": max_attempts,
    }


def sql_tool(state: AgentState) -> AgentState:
    """Run every sql-routed sub-question in the plan."""
    results = list(state.get("tool_results", []))
    attempts = state.get("sql_attempts", 0)

    targets = [step for step in state.get("plan", []) if step.get("tool") == "sql"]
    if not targets:
        targets = [{"sub_question": state["question"]}]

    # §6.5: semantic retrieval returns IDs, and SQL does the arithmetic over
    # them. Scoping the query to those IDs is what keeps the embedding layer
    # from ever producing a figure.
    retrieved: list[int] = []
    for prior in results:
        retrieved.extend(prior.get("transaction_ids") or [])

    for step in targets:
        result = run_query(step.get("sub_question", state["question"]),
                           transaction_ids=retrieved or None)
        result["sub_question"] = step.get("sub_question", state["question"])
        results.append(result)
        attempts += result["attempts"]

    return {**state, "tool_results": results, "sql_attempts": attempts}
