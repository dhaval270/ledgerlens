"""Shared fixtures.

The synthetic dataset is a build artifact, not source. Tests that need it
generate it on demand rather than assuming a previous run left it behind —
otherwise the suite passes locally and fails on a fresh checkout.
"""

from __future__ import annotations

import pytest

from ledgerlens.synthetic import OUT_DIR, write

SYNTHETIC_CSV = OUT_DIR / "transactions.csv"


@pytest.fixture(scope="session", autouse=True)
def synthetic_data():
    """Generate the synthetic dataset once per session if it is missing.

    Deterministic (seeded), so regenerating cannot change any assertion.
    """
    if not SYNTHETIC_CSV.exists():
        write()
    return SYNTHETIC_CSV
