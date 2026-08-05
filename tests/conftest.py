"""Shared fixtures.

The synthetic dataset is a build artifact, not source. Tests that need it
generate it on demand rather than assuming a previous run left it behind —
otherwise the suite passes locally and fails on a fresh checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

# Pin the suite to the synthetic benchmark ledger before anything imports
# ledgerlens.db, which resolves LEDGERLENS_DB at import time. Without this, a
# machine configured to open a personal ledger scores the golden set against
# personal transactions — which is not a failing test, it is a meaningless
# passing one. `load_dotenv` does not override an existing variable, so setting
# it here wins over .env.
os.environ["LEDGERLENS_DB"] = str(Path(__file__).resolve().parent.parent / "ledger.db")

import pytest  # noqa: E402

from ledgerlens.synthetic import OUT_DIR, write  # noqa: E402

SYNTHETIC_CSV = OUT_DIR / "transactions.csv"


@pytest.fixture(scope="session", autouse=True)
def synthetic_data():
    """Generate the synthetic dataset once per session if it is missing.

    Deterministic (seeded), so regenerating cannot change any assertion.
    """
    if not SYNTHETIC_CSV.exists():
        write()
    return SYNTHETIC_CSV
