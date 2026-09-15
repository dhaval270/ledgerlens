#!/bin/sh
# The ledger lives on a mounted volume, which is empty on the first boot of a
# new deployment. Seed it once with the synthetic set so the demo has something
# to answer questions about; every boot after this finds a database and skips.
set -e

if [ ! -f "$LEDGERLENS_DB" ]; then
    echo "no ledger at $LEDGERLENS_DB — seeding the synthetic set"
    python -m ledgerlens.synthetic
    python -m ledgerlens.ingest --init --resolve data/synthetic/transactions.csv
else
    echo "ledger found at $LEDGERLENS_DB"
fi

exec uvicorn ledgerlens.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
