"""P2P counterparty extraction — the payee on a transfer is a person.

Resolving every Zelle payment to the merchant `Zelle` collapses everyone you
have ever paid into one row, so a join cannot answer "how much did I send to X?"
and the name survives only inside `raw_descriptor`. Observed live: the model
wrote `WHERE m.canonical_name = 'Dhaval Patel'` — the right shape against a
schema where payees are merchants.

Names and confirmation codes below are invented. The parser was developed
against real Zelle statements, but a tool whose whole argument is that
financial data deserves care should not ship anyone's payment history in its
own test suite.
"""

from __future__ import annotations

import pytest

from ledgerlens.ingest.merchants import counterparty


@pytest.mark.parametrize("descriptor,expected", [
    ("Zelle Payment To Priya Raman Jpm99Aa1Bb2C", "Priya Raman"),
    ("Zelle Payment From Nadia Okonkwo 30000000000", "Nadia Okonkwo"),
    ("Zelle Payment To Anneliese Vandermolen Jpm99Gg5Hh6I", "Anneliese Vandermolen"),
    ("Zelle Payment From Dhaval Patel Wfct000X0Aaa", "Dhaval Patel"),
])
def test_extracts_the_person(descriptor, expected):
    assert counterparty(descriptor) == expected


def test_a_phone_number_recipient_keeps_its_identifier():
    """Stripping every digit-bearing token left nothing and lost the payee."""
    assert counterparty("Zelle Payment To 1400000000 29488151212") == "1400000000"


def test_other_rails_are_recognized():
    assert counterparty("Venmo Payment To Alex Smith Abc12345") == "Alex Smith"
    assert counterparty("PayPal To Jane Doe") == "Jane Doe"


def test_case_is_normalized():
    assert counterparty("ZELLE PAYMENT TO PRIYA RAMAN JPM99X1") == "Priya Raman"


# --- what it must not claim --------------------------------------------------

def test_a_card_payment_is_not_a_counterparty():
    assert counterparty("07/18 Payment To Chase Card Ending IN 2811") is None


def test_an_ordinary_merchant_is_not_a_counterparty():
    assert counterparty("TRADER JOE'S #77811 PORTLAND OR") is None


def test_payroll_is_not_a_counterparty():
    assert counterparty("Commonwealth of Um Payroll PPD ID: 04-3167352") is None


def test_a_surname_containing_no_digits_is_never_stripped():
    """The reference pattern requires a digit precisely so this cannot happen."""
    assert counterparty("Zelle Payment To Alexander Constantinou") == "Alexander Constantinou"
