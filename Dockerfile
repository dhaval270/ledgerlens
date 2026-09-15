# LedgerLens — a stateful container, not a serverless function.
#
# sentence-transformers pulls in torch (~518 MB), and the ledger is a SQLite
# file that /ingest and every approval write to. Both facts rule out the
# serverless hosts; what this needs is a container with a disk attached.

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

# The ledger lives on the mounted volume, not in the image — an image layer is
# recreated on every deploy, which would silently discard uploaded statements.
ENV LEDGERLENS_DB=/data/ledger.db \
    PYTHONUNBUFFERED=1
VOLUME /data

EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
