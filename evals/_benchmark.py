"""Pin every eval to the benchmark ledger, before `ledgerlens.db` is imported.

`DB_PATH` is resolved from `LEDGERLENS_DB` at import time, and `.env` sets that
variable for whoever is using the app on their own statements. So an eval run
from a shell configured for real data scored the golden set against a personal
ledger — 16 transactions, no anomalies, no categories. That is not a failing
benchmark, it is a meaningless passing one: `run_verifier_eval.py` crashed with
`IndexError` because the anomaly detector found nothing to detect, and the
crash was the lucky outcome. The golden queries would simply have returned
None and been scored as regressions.

`tests/conftest.py` has pinned the suite this way from the start; the evals
never did. Import this module first — before any `ledgerlens` import — in
anything that scores against `evals/golden_queries.jsonl`.
"""

from __future__ import annotations

import os
from pathlib import Path

BENCHMARK_DB = Path(__file__).resolve().parent.parent / "ledger.db"

# Set unconditionally, overriding .env and the ambient shell alike: an eval that
# quietly scores a different ledger is the failure this exists to prevent, and
# "unset it first" is not a thing anyone remembers to do.
os.environ["LEDGERLENS_DB"] = str(BENCHMARK_DB)
