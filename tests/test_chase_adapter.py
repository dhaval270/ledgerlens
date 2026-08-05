"""Chase checking PDF adapter — §5.1.

Exercised through `parse_text`, not a fixture PDF: real statements are the one
thing that can never be committed (§10). The samples below reproduce Chase's
text layer, including its `*start*`/`*end*` markers.
"""

from __future__ import annotations

import pytest

from ledgerlens.ingest.parse import ChaseCheckingPDF, StatementError

HEADER = "June 26, 2026throughJuly 24, 2026\nJPMorgan Chase Bank, N.A.\n"


def statement(rows: str, opening: str = "983.30", closing: str = "148.94") -> str:
    return (
        HEADER
        + "*start*transaction detail\n"
        "TRANSACTION DETAIL\n"
        "DATE DESCRIPTION AMOUNT BALANCE\n"
        f"Beginning Balance ${opening}\n"
        f"{rows}\n"
        f"Ending Balance ${closing}\n"
        "*end*transaction detail\n"
    )


REAL_SHAPE = (
    "06/29 Zelle Payment To Manan Parikh Jpm99Cmxs17C -146.91 836.39\n"
    "07/01 06/30 Payment To Chase Card Ending IN 2811 -205.05 631.34\n"
    "07/13 07/11 Payment To Chase Card Ending IN 2811 -207.29 424.05\n"
    "07/13 Zelle Payment To Manan Parikh Jpm99Cp3Qlzf -55.93 368.12\n"
    "07/13 Zelle Payment To Satyakeerthana Katragadda Jpm99Cp6S1Ue -5.00 363.12\n"
    "07/20 07/18 Payment To Chase Card Ending IN 2811 -222.07 141.05\n"
    "07/22 Zelle Payment From Sanchari Sadhukhan 30099853170 7.89 148.94"
)


# --- the happy path ----------------------------------------------------------

def test_parses_every_transaction_row():
    rows = ChaseCheckingPDF.parse_text(statement(REAL_SHAPE))
    assert len(rows) == 7


def test_balance_lines_are_not_transactions():
    """'Beginning Balance $983.30' must not become a row."""
    rows = ChaseCheckingPDF.parse_text(statement(REAL_SHAPE))
    assert not any("Balance" in r.descriptor for r in rows)


def test_amounts_keep_their_sign():
    rows = ChaseCheckingPDF.parse_text(statement(REAL_SHAPE))
    assert rows[0].amount == -146.91
    assert rows[-1].amount == 7.89


def test_the_balance_column_is_not_mistaken_for_the_amount():
    rows = ChaseCheckingPDF.parse_text(statement(REAL_SHAPE))
    assert rows[0].amount == -146.91      # not 836.39


def test_a_second_date_stays_in_the_descriptor():
    """'07/01 06/30 Payment To...' posts on the 1st; the 30th is Chase's own note."""
    rows = ChaseCheckingPDF.parse_text(statement(REAL_SHAPE))
    row = rows[1]
    assert row.posted_date == "2026-07-01"
    assert row.descriptor.startswith("06/30 Payment To Chase Card")


# --- years, which the rows do not carry --------------------------------------

def test_year_comes_from_the_statement_period():
    rows = ChaseCheckingPDF.parse_text(statement(REAL_SHAPE))
    assert rows[0].posted_date == "2026-06-29"


def test_a_statement_spanning_new_year_splits_the_years():
    """The failure this prevents: 12/28 and 01/03 filed twelve months apart."""
    text = (
        "December 26, 2025throughJanuary 24, 2026\nJPMorgan Chase Bank, N.A.\n"
        "*start*transaction detail\n"
        "Beginning Balance $500.00\n"
        "12/28 Zelle Payment To Someone -100.00 400.00\n"
        "01/03 Zelle Payment To Someone Else -50.00 350.00\n"
        "Ending Balance $350.00\n"
        "*end*transaction detail\n"
    )
    rows = ChaseCheckingPDF.parse_text(text)
    assert [r.posted_date for r in rows] == ["2025-12-28", "2026-01-03"]


# --- transfers are not purchases ---------------------------------------------

def test_card_payments_and_zelle_are_transfers():
    """§4 filters every analytic on type='purchase'; these would inflate spend."""
    rows = ChaseCheckingPDF.parse_text(statement(REAL_SHAPE))
    assert {r.txn_type for r in rows} == {"transfer"}


def test_payroll_is_income():
    text = statement(
        "07/01 Acme Corp Payroll Ppd Id 123 1500.00 2483.30",
        closing="2483.30",
    )
    assert ChaseCheckingPDF.parse_text(text)[0].txn_type == "income"


def test_service_fees_are_fees():
    text = statement("07/01 Monthly Service Fee -12.00 971.30", closing="971.30")
    assert ChaseCheckingPDF.parse_text(text)[0].txn_type == "fee"


def test_an_unrecognized_outflow_stays_a_purchase():
    """Conservative on purpose: visible in a total beats silently absent."""
    text = statement("07/01 Trader Joes 0421 Amherst MA -43.10 940.20", closing="940.20")
    assert ChaseCheckingPDF.parse_text(text)[0].txn_type == "purchase"


# --- the checksum ------------------------------------------------------------

def test_a_statement_that_does_not_add_up_is_refused():
    """A dropped or misread row must fail loudly, not enter the ledger."""
    dropped = "\n".join(REAL_SHAPE.splitlines()[:-1])   # lose the last row
    with pytest.raises(StatementError, match="reconciliation"):
        ChaseCheckingPDF.parse_text(statement(dropped))


def test_reconciliation_accepts_the_intact_statement():
    assert len(ChaseCheckingPDF.parse_text(statement(REAL_SHAPE))) == 7


# --- refusing what it does not understand ------------------------------------

def test_missing_period_line_is_refused():
    text = statement(REAL_SHAPE).replace("June 26, 2026throughJuly 24, 2026", "")
    with pytest.raises(StatementError, match="period"):
        ChaseCheckingPDF.parse_text(text)


def test_missing_markers_are_refused():
    text = statement(REAL_SHAPE).replace("*start*transaction detail", "")
    with pytest.raises(StatementError, match="markers"):
        ChaseCheckingPDF.parse_text(text)


def test_an_empty_table_is_refused_not_returned_as_success():
    """A silent [] reads as 'no transactions', indistinguishable from success."""
    with pytest.raises(StatementError, match="zero rows"):
        ChaseCheckingPDF.parse_text(statement("TRANSACTION DETAIL"))


# --- dispatch ----------------------------------------------------------------

def test_matches_a_chase_pdf():
    from pathlib import Path
    assert ChaseCheckingPDF.matches(Path("s.pdf"), statement(REAL_SHAPE))


def test_does_not_match_a_csv_or_another_bank():
    from pathlib import Path
    assert not ChaseCheckingPDF.matches(Path("s.csv"), statement(REAL_SHAPE))
    assert not ChaseCheckingPDF.matches(Path("s.pdf"), "Bank of America\nstatement")
