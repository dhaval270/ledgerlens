"""Verifier tests — §6.7. Pure functions, no network.

§9.3 wants a *verifier catch rate*: inject wrong numbers, confirm rejection.
These are that experiment in unit form.
"""

from __future__ import annotations

from ledgerlens.agent.nodes.verifier import extract_numerics, verify


def rows(*values: dict) -> list[dict]:
    return [{"tool": "sql", "query": "SELECT 1", "rows": list(values), "error": None}]


# --- numeric extraction ------------------------------------------------------

def test_extracts_currency_and_separators():
    got = extract_numerics("You spent $1,234.56 and $78.90 last month.")
    assert 1234.56 in got and 78.90 in got


def test_extracts_negatives_and_percentages():
    got = extract_numerics("Down -412.55, which is 53.5% of the total.")
    assert 412.55 in [abs(g) for g in got]
    assert 53.5 in got


# --- the core claim: unsupported numbers are rejected ------------------------

def test_supported_figure_passes():
    v = verify("You spent $412.55 on groceries.", rows({"total": -412.55}))
    assert v["pass"], v["reason"]


def test_hallucinated_figure_is_rejected():
    """The §9.3 catch-rate experiment: the tool said 412.55, the answer says 500."""
    v = verify("You spent $500.00 on groceries.", rows({"total": -412.55}))
    assert not v["pass"]
    assert 500.0 in v["unsupported"]


def test_one_wrong_figure_among_correct_ones_is_caught():
    v = verify(
        "Groceries were $412.55, dining $88.20, and travel $999.99.",
        rows({"groceries": -412.55}, {"dining": -88.20}, {"travel": -150.00}),
    )
    assert not v["pass"]
    assert 999.99 in v["unsupported"]


def test_sign_flip_is_accepted():
    """Outflows are stored negative; rendering them positive in prose is correct."""
    assert verify("You spent $412.55.", rows({"total": -412.55}))["pass"]


def test_rounding_within_tolerance_is_accepted():
    assert verify("About 53.5% of spending.", rows({"pct": 53.52333}))["pass"]


def test_small_integers_are_not_treated_as_claims():
    """'the top 3 merchants' is prose, not a figure to audit."""
    assert verify("Your top 3 merchants by spend totalled $412.55.",
                  rows({"total": -412.55}))["pass"]


# --- empty and error states --------------------------------------------------

def test_no_tool_results_fails():
    v = verify("You spent $412.55.", [])
    assert not v["pass"]
    assert "no tool results" in v["reason"]


def test_empty_rows_cannot_support_a_figure():
    v = verify("You spent $412.55.", [{"tool": "sql", "rows": [], "error": None}])
    assert not v["pass"]


def test_answer_with_no_numbers_passes():
    v = verify("I could not find any matching transactions.", rows({"total": -1.0}))
    assert v["pass"]
    assert v["checked"] == 0


# --- date range --------------------------------------------------------------

def test_claim_outside_retrieved_range_is_rejected():
    v = verify(
        "In 2019 you spent $412.55.",
        rows({"d": "2026-03-01", "total": -412.55}, {"d": "2026-03-31", "total": -412.55}),
    )
    assert not v["pass"]
    assert "2019" in v["reason"]


def test_claim_inside_retrieved_range_is_accepted():
    v = verify(
        "On 2026-03-15 you spent $412.55.",
        rows({"d": "2026-03-01", "total": -412.55}, {"d": "2026-03-31", "total": -412.55}),
    )
    assert v["pass"], v["reason"]


# --- the failure mode this node exists for -----------------------------------

def test_catches_a_confident_wrong_answer_that_repair_would_miss():
    """Valid SQL, clean execution, wrong number — the dominant golden-set failure.

    agg-04 returned -4.38 (MAX of a negative column) when the true largest
    purchase was -1450.00. Nothing errored, so the repair loop stayed silent.
    """
    v = verify("Your largest purchase was $1,450.00.", rows({"largest": -4.38}))
    assert not v["pass"]
    assert 1450.0 in v["unsupported"]
