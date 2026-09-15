#!/bin/sh
# The image ships with a ledger already built (see Dockerfile), so the common
# path here does nothing. The seed stays as a fallback for a deployment that
# mounts an empty volume over /app or points LEDGERLENS_DB somewhere new —
# better a slow first boot than a service answering every question with
# "no matching transactions".
set -e

if [ ! -f "$LEDGERLENS_DB" ]; then
    echo "no ledger at $LEDGERLENS_DB — seeding"
    python -m ledgerlens.synthetic
    python -m ledgerlens.ingest --init data/synthetic/transactions.csv
    python -m ledgerlens.index
else
    echo "ledger found at $LEDGERLENS_DB"
fi

exec uvicorn ledgerlens.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
