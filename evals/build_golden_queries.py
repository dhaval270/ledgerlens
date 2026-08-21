"""Build the golden query set — §9.1.

Expected values are *computed* by running reference SQL against the populated
database, never hand-written. Hand-typed figures rot the moment the generator
changes and quietly turn a passing eval into a lie; computed ones stay true
because they are re-derived from whatever is actually in the ledger.

**The ledger itself is not reproducible.** The transaction generator is seeded,
but tier-3 merchant resolution calls an LLM, so rebuilding from the same CSV
does not produce the same `merchants` table: two consecutive rebuilds named the
same airline `Delta` and then `Delta Air Lines`, silently breaking a reference
SQL that matched on equality. Reference SQL must therefore match merchant names
by prefix (`LIKE 'Delta%'`), never by `=`, unless the name comes from a
deterministic tier. Nothing downstream can assume a stable merchant vocabulary,
including any figure quoted from a previous run.

The reference SQL is the answer key, not a template the agent must reproduce.
Execution accuracy compares final values, so any SQL reaching the same number
counts as correct — which is the right bar for §9.3.

    python evals/build_golden_queries.py

Ledger window: 2025-08-01 .. 2026-07-31.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # evals/ is not a package

from ledgerlens.db import DB_PATH, connect_readonly

OUT = Path(__file__).parent / "golden_queries.jsonl"

PURCHASE = "t.type = 'purchase'"


# --- detector-backed answer keys ---------------------------------------------
# SQLite has no STDDEV, and §6.6's baseline is leave-one-out over a trailing
# window — procedural, not expressible as a scalar SELECT. Rather than hand-type
# the figures (which is exactly what this file exists to avoid), these read the
# detector itself. The invariant holds: expected values are still *computed*,
# and `test_expected_values_still_match_reference_sql` re-runs them the same way.
#
# This makes the detector its own answer key, so these queries cannot catch a
# detector that is wrong — only one that has *changed*. What they do test is the
# agent path: whether the planner routes here at all, and whether the findings
# survive to the answer intact.

def _findings(conn: sqlite3.Connection):
    from ledgerlens.agent.nodes.anomaly_tool import detect

    return detect(conn)


def top_anomaly_category(conn):
    return _findings(conn)[0].category


def top_anomaly_period(conn):
    return _findings(conn)[0].period


def top_anomaly_observed(conn):
    return _findings(conn)[0].observed


def top_anomaly_baseline(conn):
    return _findings(conn)[0].baseline_mean


def top_decline_category(conn):
    return next(f.category for f in _findings(conn) if f.direction == "below")


def anomaly_count(conn):
    return len(_findings(conn))


DETECTORS = {fn.__name__: fn for fn in (
    top_anomaly_category, top_anomaly_period, top_anomaly_observed,
    top_anomaly_baseline, top_decline_category, anomaly_count,
)}


# (id, question, tools, reference, note)
# `reference` is SQL returning exactly one scalar, a callable from DETECTORS,
# or None to mark a refusal case.
QUERIES: list[tuple[str, str, list[str], str | None, str]] = [

    # --- simple aggregates -------------------------------------------------
    ("agg-01", "How much did I spend on groceries in March 2026?", ["sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE c.name='groceries' AND {PURCHASE} AND substr(t.posted_date,1,7)='2026-03'""",
     "single category, single month"),

    ("agg-02", "What was my total spending in January 2026?", ["sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t
         WHERE {PURCHASE} AND substr(t.posted_date,1,7)='2026-01'""",
     "all purchases in a month"),

    ("agg-03", "How many transactions do I have in total?", ["sql"],
     "SELECT COUNT(*) FROM transactions t", "raw count, all types"),

    ("agg-04", "What was my single largest purchase?", ["sql"],
     f"SELECT ROUND(MIN(t.amount),2) FROM transactions t WHERE {PURCHASE}",
     "largest outflow = most negative"),

    ("agg-05", "How much did I spend on coffee over the whole period?", ["sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE c.name='coffee' AND {PURCHASE}""", "category total"),

    ("agg-06", "How many times did I buy coffee?", ["sql"],
     f"""SELECT COUNT(*) FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE c.name='coffee' AND {PURCHASE}""", "count not sum"),

    ("agg-07", "What is my average grocery transaction?", ["sql"],
     f"""SELECT ROUND(AVG(t.amount),2) FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE c.name='groceries' AND {PURCHASE}""", "average"),

    ("agg-08", "How much total income did I receive?", ["sql"],
     "SELECT ROUND(SUM(t.amount),2) FROM transactions t WHERE t.type='income'",
     "income is inflow, positive"),

    ("agg-09", "How much did I pay in bank fees?", ["sql"],
     "SELECT ROUND(SUM(t.amount),2) FROM transactions t WHERE t.type='fee'",
     "non-purchase type"),

    ("agg-10", "How much did I spend on rent in total?", ["sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE c.name='rent' AND {PURCHASE}""", "large recurring category"),

    # --- date ranges --------------------------------------------------------
    ("date-01", "How much did I spend between 2026-01-01 and 2026-03-31?", ["sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t
         WHERE {PURCHASE} AND t.posted_date BETWEEN '2026-01-01' AND '2026-03-31'""",
     "explicit range"),

    ("date-02", "What did I spend in the last three months of the data?", ["sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t
         WHERE {PURCHASE} AND t.posted_date >= '2026-05-01'""", "relative window"),

    ("date-03", "How much did I spend on dining in the second half of 2025?", ["sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE c.name='dining' AND {PURCHASE} AND t.posted_date < '2026-01-01'""",
     "category + range"),

    ("date-04", "What is the date of my earliest transaction?", ["sql"],
     "SELECT MIN(t.posted_date) FROM transactions t", "date scalar"),

    ("date-05", "What is the date of my most recent transaction?", ["sql"],
     "SELECT MAX(t.posted_date) FROM transactions t", "date scalar"),

    ("date-06", "How much did I spend on travel in April 2026?", ["sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE c.name='travel' AND {PURCHASE} AND substr(t.posted_date,1,7)='2026-04'""",
     "the injected spike month"),

    ("date-07", "How many months of data do I have?", ["sql"],
     "SELECT COUNT(DISTINCT substr(t.posted_date,1,7)) FROM transactions t", "distinct months"),

    ("date-08", "How much did I spend in December 2025?", ["sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t
         WHERE {PURCHASE} AND substr(t.posted_date,1,7)='2025-12'""", "single month"),

    # --- comparisons (multi-step) -------------------------------------------
    ("cmp-01", "Did I spend more on groceries or dining overall?", ["sql"],
     """SELECT c.name FROM transactions t JOIN categories c ON c.id=t.category_id
        WHERE c.name IN ('groceries','dining') AND t.type='purchase'
        GROUP BY c.name ORDER BY SUM(t.amount) ASC LIMIT 1""",
     "two aggregates then compare; answer is a name"),

    ("cmp-02", "How much more did I spend in April 2026 than in March 2026?", ["sql"],
     f"""SELECT ROUND(
           (SELECT SUM(t.amount) FROM transactions t WHERE {PURCHASE} AND substr(t.posted_date,1,7)='2026-04')
         - (SELECT SUM(t.amount) FROM transactions t WHERE {PURCHASE} AND substr(t.posted_date,1,7)='2026-03'), 2)""",
     "difference of two windows"),

    ("cmp-03", "Which month had the highest total spending?", ["sql"],
     f"""SELECT substr(t.posted_date,1,7) FROM transactions t WHERE {PURCHASE}
         GROUP BY 1 ORDER BY SUM(t.amount) ASC LIMIT 1""", "argmax over months"),

    ("cmp-04", "Which category do I spend the most on?", ["sql"],
     f"""SELECT c.name FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE {PURCHASE} GROUP BY c.id ORDER BY SUM(t.amount) ASC LIMIT 1""", "argmax"),

    ("cmp-05", "Which merchant did I spend the most money at?", ["sql"],
     f"""SELECT m.canonical_name FROM transactions t JOIN merchants m ON m.id=t.merchant_id
         WHERE {PURCHASE} GROUP BY m.id ORDER BY SUM(t.amount) ASC LIMIT 1""", "argmax over merchants"),

    ("cmp-06", "Which merchant do I visit most often?", ["sql"],
     f"""SELECT m.canonical_name FROM transactions t JOIN merchants m ON m.id=t.merchant_id
         WHERE {PURCHASE} GROUP BY m.id ORDER BY COUNT(*) DESC LIMIT 1""",
     "frequency not value — a classic confusion"),

    ("cmp-07", "Was my spending in the first six months higher than the last six?", ["sql"],
     f"""SELECT CASE WHEN
           (SELECT SUM(t.amount) FROM transactions t WHERE {PURCHASE} AND t.posted_date < '2026-02-01')
         < (SELECT SUM(t.amount) FROM transactions t WHERE {PURCHASE} AND t.posted_date >= '2026-02-01')
         THEN 'yes' ELSE 'no' END""", "boolean comparison"),

    ("cmp-08", "How much less did I spend on dining than on groceries?", ["sql"],
     """SELECT ROUND(
          (SELECT SUM(t.amount) FROM transactions t JOIN categories c ON c.id=t.category_id
           WHERE c.name='dining' AND t.type='purchase')
        - (SELECT SUM(t.amount) FROM transactions t JOIN categories c ON c.id=t.category_id
           WHERE c.name='groceries' AND t.type='purchase'), 2)""", "signed difference"),

    # --- merchant-specific --------------------------------------------------
    ("mer-01", "How much did I spend at Amazon?", ["sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
         WHERE m.canonical_name LIKE 'Amazon%' AND {PURCHASE}""",
     "the merchant that was split before the tier-3 fix"),

    ("mer-02", "How many times did I shop at Target?", ["sql"],
     f"""SELECT COUNT(*) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
         WHERE m.canonical_name LIKE 'Target%' AND {PURCHASE}""", "merchant count"),

    ("mer-03", "How much do I pay Netflix each month?", ["sql"],
     """SELECT ROUND(s.typical_amount,2) FROM recurring_series s
        JOIN merchants m ON m.id=s.merchant_id WHERE m.canonical_name LIKE 'Netflix%'""",
     "reads recurring_series, not raw transactions"),

    ("mer-04", "How many distinct merchants have I bought from?", ["sql"],
     f"""SELECT COUNT(DISTINCT t.merchant_id) FROM transactions t
         WHERE t.merchant_id IS NOT NULL AND {PURCHASE}""",
     "distinct count; 'bought from' excludes payroll and transfers"),

    ("mer-05", "What did I spend at Whole Foods?", ["sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
         WHERE m.canonical_name LIKE 'Whole Foods%' AND {PURCHASE}""", "abbreviation-resolved merchant"),

    ("mer-06", "How much did I spend on Uber rides?", ["sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
         WHERE m.canonical_name LIKE 'Uber%' AND {PURCHASE}""", "merchant total"),

    # --- recurring / anomaly ------------------------------------------------
    ("rec-01", "How many recurring charges do I have?", ["sql"],
     "SELECT COUNT(*) FROM recurring_series s WHERE s.active=1", "recurring_series count"),

    ("rec-02", "Did any of my subscriptions go up in price?", ["anomaly", "sql"],
     """SELECT m.canonical_name FROM recurring_series s JOIN merchants m ON m.id=s.merchant_id
        WHERE ABS(s.last_amount) > ABS(s.typical_amount)*1.05 LIMIT 1""",
     "the injected Spotify hike"),

    ("rec-03", "What is my most expensive recurring charge?", ["sql"],
     """SELECT m.canonical_name FROM recurring_series s JOIN merchants m ON m.id=s.merchant_id
        WHERE s.typical_amount < 0 ORDER BY s.typical_amount ASC LIMIT 1""", "argmax over series"),

    ("rec-04", "How much do my subscriptions cost me per month in total?", ["sql"],
     """SELECT ROUND(SUM(s.typical_amount),2) FROM recurring_series s
        JOIN merchants m ON m.id=s.merchant_id JOIN categories c ON c.id=m.default_category_id
        WHERE s.active=1 AND c.name='subscriptions'""", "sum over a filtered series set"),

    ("rec-05", "Is my travel spending unusual in any month?", ["anomaly"],
     f"""SELECT substr(t.posted_date,1,7) FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE c.name='travel' AND {PURCHASE} GROUP BY 1 ORDER BY SUM(ABS(t.amount)) DESC LIMIT 1""",
     "the injected z=16 spike"),

    ("rec-06", "When did I first shop at Peloton?", ["sql"],
     """SELECT MIN(t.posted_date) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
        WHERE m.canonical_name LIKE 'Peloton%'""", "new merchant, final month only"),

    # --- multi-hop ----------------------------------------------------------
    ("multi-01", "What percentage of my spending was on essentials?", ["sql"],
     f"""SELECT ROUND(100.0 *
           (SELECT SUM(t.amount) FROM transactions t JOIN categories c ON c.id=t.category_id
            WHERE c.is_essential=1 AND {PURCHASE})
         / (SELECT SUM(t.amount) FROM transactions t WHERE {PURCHASE}), 1)""",
     "ratio of two aggregates"),

    ("multi-02", "In my highest-spending month, which category dominated?", ["sql"],
     f"""SELECT c.name FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE {PURCHASE} AND substr(t.posted_date,1,7) = (
           SELECT substr(t2.posted_date,1,7) FROM transactions t2 WHERE t2.type='purchase'
           GROUP BY 1 ORDER BY SUM(t2.amount) ASC LIMIT 1)
         GROUP BY c.id ORDER BY SUM(t.amount) ASC LIMIT 1""",
     "argmax feeding a second argmax"),

    ("multi-03", "What is my average monthly spending?", ["sql"],
     f"""SELECT ROUND(AVG(monthly),2) FROM (
           SELECT SUM(t.amount) AS monthly FROM transactions t WHERE {PURCHASE}
           GROUP BY substr(t.posted_date,1,7))""", "aggregate of an aggregate"),

    ("multi-04", "How much did I save, counting income minus spending?", ["sql"],
     "SELECT ROUND(SUM(t.amount),2) FROM transactions t WHERE t.type IN ('income','purchase','fee')",
     "mixed types, sign-aware"),

    ("multi-05", "What percentage of my grocery spending went to Trader Joe's?", ["sql"],
     """SELECT ROUND(100.0 *
          (SELECT SUM(t.amount) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
           WHERE m.canonical_name LIKE 'Trader Joe%' AND t.type='purchase')
        / (SELECT SUM(t.amount) FROM transactions t JOIN categories c ON c.id=t.category_id
           WHERE c.name='groceries' AND t.type='purchase'), 1)""", "merchant share of category"),

    # --- semantic recall ----------------------------------------------------
    ("sem-01", "What did I spend on anything medical-looking?", ["semantic", "sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE c.name='health' AND {PURCHASE}""", "fuzzy phrasing, health category"),

    ("sem-02", "How much went on entertainment and going out?", ["semantic", "sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE c.name IN ('entertainment','dining') AND {PURCHASE}""", "spans two categories"),

    ("sem-03", "What did my trips cost me?", ["semantic", "sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN categories c ON c.id=t.category_id
         WHERE c.name='travel' AND {PURCHASE}""", "colloquial phrasing"),

    # Each of the below names neither a category nor a merchant, so the wording
    # alone cannot be matched by a literal filter — which is the condition §6.3
    # gives for routing to `semantic` at all.
    #
    # Every target is deliberately under TOP_K (20) transactions. §6.5 scopes the
    # SQL to the IDs retrieval returned, so a question whose true answer spans
    # more rows than retrieval returns is unanswerable *by construction* through
    # this path — the ceiling is the retrieval depth, not the model. sem-01..03
    # above sit on the wrong side of that line (health is 52 rows, travel 21),
    # which is worth knowing and is why these were added rather than replacing
    # them.

    ("sem-04", "What do I pay for my gym membership?", ["semantic", "sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
         WHERE m.canonical_name LIKE 'Planet Fit%' AND {PURCHASE}""",
     "13 rows; 'gym' appears in no field — descriptor reads PLANET FIT"),

    ("sem-05", "How much did that hotel stay in Austin cost me?", ["semantic", "sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
         WHERE m.canonical_name LIKE 'Marriott%' AND {PURCHASE}""",
     "8 rows; the city is only in raw_descriptor, so recall must read it"),

    ("sem-06", "How much have I spent on flights?", ["semantic", "sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
         WHERE m.canonical_name LIKE 'Delta%' AND {PURCHASE}""",
     "13 rows; category is 'travel', which also holds hotels — 'flights' is narrower"),

    ("sem-07", "What is my creative software subscription costing me?", ["semantic", "sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
         WHERE m.canonical_name LIKE 'Adobe%' AND {PURCHASE}""",
     "12 rows; one of several merchants inside 'subscriptions'"),

    ("sem-08", "How much do I pay for internet and TV?", ["semantic", "sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
         WHERE m.canonical_name LIKE 'Comcast%' AND {PURCHASE}""",
     "12 rows; descriptor is COMCAST *XFINITY, neither word in the question"),

    ("sem-09", "What did that exercise bike cost?", ["semantic", "sql"],
     f"""SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
         WHERE m.canonical_name LIKE 'Peloton%' AND {PURCHASE}""",
     "a single row — the hardest possible retrieval target"),

    ("sem-10", "How much have I lost to cash machine charges?", ["semantic", "sql"],
     """SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN merchants m ON m.id=t.merchant_id
        WHERE m.canonical_name LIKE 'ATM Fee%'""",
     "3 rows, and type='fee' — a purchase filter here returns nothing"),

    # --- anomaly routing ----------------------------------------------------
    # §6.6 is reachable from the graph but was never scored: the golden harness
    # called the SQL tool directly, so `anomaly` had appeared in two `tools`
    # fields for months without the node ever executing under evaluation.
    #
    # Note what the detector returns — every finding, ranked, ignoring the
    # sub-question — and what the answer node does with it: renders the first
    # five without computing. Questions needing arithmetic *over* findings, or a
    # finding ranked sixth or lower, are outside what this path can express.
    # anom-06 is included precisely because it asks for a count.

    ("anom-01", "Which category had the most unusual month?", ["anomaly"],
     top_anomaly_category, "the top-ranked finding by severity"),

    ("anom-02", "In which month did my travel spending look abnormal?", ["anomaly"],
     top_anomaly_period, "period of the top finding — the injected spike"),

    ("anom-03", "How much did I actually spend in that abnormal travel month?", ["anomaly"],
     top_anomaly_observed, "observed magnitude, positive — must survive to the answer"),

    ("anom-04", "What is my normal monthly travel spend, ignoring the spike?", ["anomaly"],
     top_anomaly_baseline, "the leave-one-out baseline; a plain AVG would differ"),

    ("anom-05", "Did my spending drop unusually low in any category?", ["anomaly"],
     top_decline_category, "direction='below' — decreases are anomalies too"),

    ("anom-06", "How many category-months look statistically unusual?", ["anomaly"],
     anomaly_count, "a count over findings; the answer node renders 5 and computes nothing"),

    # --- deliberately unanswerable: the correct answer is a refusal ---------
    ("refuse-01", "What will I spend next month?", [], None,
     "no forecasting capability; must refuse rather than extrapolate"),

    ("refuse-02", "What is my credit score?", [], None,
     "not derivable from statements"),

    ("refuse-03", "What is my current account balance?", [], None,
     "no balance column; statements record flows, not position"),

    ("refuse-04", "Should I invest in index funds?", [], None,
     "financial advice — explicitly out of scope per §1"),

    ("refuse-05", "How much did I spend on groceries in 2019?", [], None,
     "outside the retrieved date range; must not answer 0"),

    ("refuse-06", "What is my employer's annual revenue?", [], None,
     "not in the data at all"),

    ("refuse-07", "Am I on track to retire comfortably?", [], None,
     "advisory, and unanswerable from 12 months of flows"),
]


def build(conn: sqlite3.Connection) -> list[dict]:
    entries, failures = [], []

    for qid, question, tools, reference, note in QUERIES:
        entry = {
            "id": qid,
            "question": question,
            "tools": tools,
            "note": note,
            "expect_refusal": reference is None,
        }

        if reference is None:
            entry["expected_value"] = None
        elif callable(reference):
            try:
                value = reference(conn)
            except Exception as exc:
                failures.append((qid, f"{reference.__name__}: {exc}"))
                continue
            if value is None:
                failures.append((qid, f"{reference.__name__} returned None"))
                continue
            entry["expected_value"] = value
            entry["reference_fn"] = reference.__name__
        else:
            try:
                row = conn.execute(reference).fetchone()
            except sqlite3.Error as exc:
                failures.append((qid, str(exc)))
                continue
            if row is None or row[0] is None:
                failures.append((qid, "reference SQL returned NULL"))
                continue
            entry["expected_value"] = row[0]
            entry["reference_sql"] = " ".join(reference.split())

        entries.append(entry)

    if failures:
        print("REFERENCE SQL FAILURES:", file=sys.stderr)
        for qid, msg in failures:
            print(f"  {qid}: {msg}", file=sys.stderr)

    return entries


def main() -> int:
    if not DB_PATH.exists():
        print("ledger.db not found — run: python -m ledgerlens.ingest --reset --resolve "
              "data/synthetic/transactions.csv", file=sys.stderr)
        return 1

    with connect_readonly() as conn:
        entries = build(conn)

    with OUT.open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")

    answerable = [e for e in entries if not e["expect_refusal"]]
    refusals = [e for e in entries if e["expect_refusal"]]
    tool_mix: dict[str, int] = {}
    for e in answerable:
        tool_mix[",".join(e["tools"])] = tool_mix.get(",".join(e["tools"]), 0) + 1

    print(f"wrote {len(entries)} queries -> {OUT}")
    print(f"  answerable {len(answerable)} | refusals {len(refusals)}")
    for mix, n in sorted(tool_mix.items()):
        print(f"  tools[{mix}] {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
