"""SQL tool safety screen — §6.4. No network: these test the guard, not the model."""

from __future__ import annotations

import sqlite3

import pytest

from ledgerlens.agent.nodes.sql_tool import is_safe
from ledgerlens.db import DB_PATH, connect_readonly


# --- must be rejected --------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "DROP TABLE transactions",
    "DELETE FROM transactions",
    "UPDATE transactions SET amount = 0",
    "INSERT INTO transactions (id) VALUES (1)",
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "PRAGMA table_info(transactions)",
    "CREATE TABLE x (id INT)",
    "ALTER TABLE transactions ADD COLUMN x INT",
    "VACUUM",
])
def test_mutating_statements_are_rejected(sql):
    ok, reason = is_safe(sql)
    assert not ok and reason


def test_stacked_statement_is_rejected():
    ok, reason = is_safe("SELECT 1; DROP TABLE transactions")
    assert not ok
    assert "multiple statements" in reason


def test_hidden_mutation_after_a_comment_is_rejected():
    ok, _ = is_safe("SELECT 1 -- harmless\nDROP TABLE transactions")
    assert not ok


def test_non_select_is_rejected():
    ok, reason = is_safe("EXPLAIN SELECT * FROM transactions")
    assert not ok
    assert "SELECT or WITH" in reason


def test_empty_is_rejected():
    assert not is_safe("   ")[0]


# --- must be allowed ---------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT SUM(amount) FROM transactions WHERE type = 'purchase'",
    "WITH m AS (SELECT 1 AS x) SELECT x FROM m",
    "SELECT * FROM transactions LIMIT 5",
    "select count(*) from merchants",
])
def test_read_only_queries_are_allowed(sql):
    ok, reason = is_safe(sql)
    assert ok, reason


def test_forbidden_word_inside_a_string_literal_is_allowed():
    """A naive substring blocklist rejects this; a real merchant could be named it."""
    ok, reason = is_safe(
        "SELECT * FROM merchants WHERE canonical_name = 'Bed Bath & Update'"
    )
    assert ok, reason


def test_forbidden_word_inside_a_comment_is_allowed():
    ok, reason = is_safe("SELECT 1 /* do not DELETE this note */")
    assert ok, reason


def test_trailing_semicolon_is_tolerated():
    ok, reason = is_safe("SELECT COUNT(*) FROM transactions;")
    assert ok, reason


# --- the guard that actually matters ----------------------------------------

@pytest.mark.skipif(not DB_PATH.exists(), reason="ledger.db not built")
def test_readonly_connection_blocks_mutation_even_if_the_screen_is_bypassed():
    """The screen is for messages; this is the real guarantee."""
    with connect_readonly() as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM transactions")


# --- the prompt follows the data, not an assumption ---------------------------

def test_purchase_filter_is_suppressed_when_there_are_none(tmp_path):
    """Observed live: "who did I pay the most?" filtered `type='purchase'` on a
    checking ledger holding only transfers, and returned nothing."""
    import sqlite3

    from ledgerlens.agent.nodes.sql_tool import _spend_rule, _type_mix
    from ledgerlens.db import SCHEMA_PATH

    conn = sqlite3.connect(tmp_path / "t.db")
    conn.executescript(SCHEMA_PATH.read_text())
    conn.execute("INSERT INTO accounts (id, name) VALUES (1, 'T')")
    for i, kind in enumerate(("transfer", "transfer", "income")):
        conn.execute(
            """INSERT INTO transactions (account_id, posted_date, amount, raw_descriptor,
                                         type, source_file, content_hash, created_at)
               VALUES (1, '2026-07-01', -10, 'd', ?, 'f', ?, 'x')""", (kind, f"h{i}"))
    conn.commit()

    assert "NO `purchase` rows" in _spend_rule(conn)
    assert "transfer (2)" in _type_mix(conn)
    conn.close()


def test_purchase_filter_is_kept_when_purchases_exist(tmp_path):
    import sqlite3

    from ledgerlens.agent.nodes.sql_tool import _spend_rule
    from ledgerlens.db import SCHEMA_PATH

    conn = sqlite3.connect(tmp_path / "t.db")
    conn.executescript(SCHEMA_PATH.read_text())
    conn.execute("INSERT INTO accounts (id, name) VALUES (1, 'T')")
    conn.execute(
        """INSERT INTO transactions (account_id, posted_date, amount, raw_descriptor,
                                     type, source_file, content_hash, created_at)
           VALUES (1, '2026-07-01', -10, 'd', 'purchase', 'f', 'h', 'x')""")
    conn.commit()

    assert "filter `type = 'purchase'`" in _spend_rule(conn)
    conn.close()


# --- semantic → sql handoff --------------------------------------------------

def _ledger_with(tmp_path, rows):
    """A ledger holding (merchant, category, amount) rows. Returns (path, ids)."""
    import sqlite3

    from ledgerlens.db import SCHEMA_PATH

    path = tmp_path / "scope.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_PATH.read_text())
    conn.execute("INSERT INTO accounts (id, name) VALUES (1, 'T')")

    ids, seen = [], {}
    for i, (merchant, category, amount) in enumerate(rows):
        for table, name in (("merchants", merchant), ("categories", category)):
            if (table, name) not in seen:
                column = "canonical_name" if table == "merchants" else "name"
                cur = conn.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (name,))
                seen[(table, name)] = cur.lastrowid
        cur = conn.execute(
            """INSERT INTO transactions (account_id, posted_date, amount, raw_descriptor,
                                         merchant_id, category_id, type, source_file,
                                         content_hash, created_at)
               VALUES (1, '2026-07-01', ?, ?, ?, ?, 'purchase', 'f', ?, 'x')""",
            (amount, merchant, seen[("merchants", merchant)],
             seen[("categories", category)], f"h{i}"))
        ids.append(cur.lastrowid)
    conn.commit()
    conn.close()
    return path, ids


def test_the_hint_names_what_was_retrieved(tmp_path):
    from ledgerlens.agent.nodes.sql_tool import _scope_hint

    path, ids = _ledger_with(tmp_path, [
        ("Planet Fitness", "health", -24.99),
        ("Planet Fitness", "health", -24.99),
        ("CVS", "health", -18.40),
    ])
    hint = _scope_hint(ids, db_path=path)
    assert "Planet Fitness" in hint and "CVS" in hint and "health" in hint


def test_the_hint_tells_the_model_not_to_filter_by_id(tmp_path):
    """The ceiling this replaced.

    §6.5 retrieves TOP_K = 20 rows. "Anything medical-looking" is a total over
    52 health transactions in the benchmark ledger, so a query restricted to
    the retrieved ids computes a correct SUM of the wrong set — and passes
    verification, because every figure did come from a retrieved row. The
    routing eval made this visible the moment routing started working: the
    questions that reached semantic+sql failed and the ones that fell through
    to sql alone passed.
    """
    from ledgerlens.agent.nodes.sql_tool import _scope_hint

    path, ids = _ledger_with(tmp_path, [("CVS", "health", -18.40)])
    hint = _scope_hint(ids, db_path=path)
    assert "over" in hint and "the whole table" in hint
    assert "t.id IN" not in hint


def test_no_retrieval_means_no_hint(tmp_path):
    """An unscoped question must get the plain prompt, not an empty preamble."""
    from ledgerlens.agent.nodes.sql_tool import _scope_hint

    assert _scope_hint(None) == ""
    assert _scope_hint([]) == ""


def test_ids_that_resolve_to_nothing_add_nothing(tmp_path):
    """Uncategorized rows with no merchant would otherwise produce a hint that
    names no filter at all — worse than silence, because it still tells the
    model a search happened."""
    import sqlite3

    from ledgerlens.agent.nodes.sql_tool import _scope_hint
    from ledgerlens.db import SCHEMA_PATH

    path = tmp_path / "bare.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_PATH.read_text())
    conn.execute("INSERT INTO accounts (id, name) VALUES (1, 'T')")
    cur = conn.execute(
        """INSERT INTO transactions (account_id, posted_date, amount, raw_descriptor,
                                     type, source_file, content_hash, created_at)
           VALUES (1, '2026-07-01', -10, 'd', 'purchase', 'f', 'h', 'x')""")
    conn.commit()
    row_id = cur.lastrowid
    conn.close()

    assert _scope_hint([row_id], db_path=path) == ""


def test_the_hint_does_not_hand_over_a_ready_made_id_list(tmp_path):
    """Measured, not assumed.

    An earlier version passed the ids alongside an instruction to prefer the
    category or merchant. The instruction lost: asked what it spent on anything
    medical-looking, the model wrote `WHERE id IN (695, 412, 488, ...)` over the
    twenty ids it had been given and returned -805.93 against a true -1874.81 —
    verified, because every figure came from a retrieved row.
    """
    from ledgerlens.agent.nodes.sql_tool import _scope_hint

    path, ids = _ledger_with(tmp_path, [
        ("CVS", "health", -18.40), ("Planet Fitness", "health", -24.99),
    ])
    hint = _scope_hint(ids, db_path=path)
    for transaction_id in ids:
        assert str(transaction_id) not in hint
