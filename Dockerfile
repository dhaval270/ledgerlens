# LedgerLens — a container, not a serverless function.
#
# sentence-transformers pulls in torch (~518 MB), against a 250 MB limit on the
# serverless hosts, and paused approvals live in an in-memory checkpointer that
# a second invocation could not see. Both rule out that shape of deployment.

FROM python:3.12-slim

# pdfplumber needs no system packages, but torch wheels want libgomp.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only torch build first: the default wheel bundles CUDA and is
# several GB for hardware no container host here will give you.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image. Downloaded at first use instead, it
# is a ~90 MB fetch on the first question after every cold start, and on a host
# with a read-only image layer it may not be cacheable at all.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY ledgerlens/ ./ledgerlens/
COPY docker-entrypoint.sh .

ENV LEDGERLENS_DB=/app/ledger.db \
    PYTHONUNBUFFERED=1

# Bake the demo ledger and its vector index into the image rather than building
# them on boot. A free-tier instance sleeps when idle and cold-starts often, and
# generating 832 rows and embedding them takes far longer than a visitor will
# wait. Built once here, every boot after is instant.
#
# Deliberately no --resolve: tier-3 merchant resolution is an LLM call, so it
# would need an API key at build time and would spend quota on every image
# rebuild. Tiers 1 and 2 resolve 96.9% of descriptors with no model at all,
# which is plenty for a demo.
RUN python -m ledgerlens.synthetic \
    && python -m ledgerlens.ingest --init data/synthetic/transactions.csv \
    && python -m ledgerlens.index

EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
