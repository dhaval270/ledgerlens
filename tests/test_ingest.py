"""Ingestion tests — §5.1/§5.2.

The dedup test is the load-bearing one: manual upload means the same statement
gets uploaded repeatedly, and the idempotency claim has to be real.
"""

from __future__ import annotations

import sqlite3

import pytest

from ledgerlens.db import SCHEMA_PATH
from ledgerlens.ingest import ingest_file
from ledgerlens.ingest.dedup import content_hash
from ledgerlens.ingest.parse import (
    NoTextLayer,
    SyntheticCSV,
    UnknownFormat,
    detect_adapter,
    parse,
)
from ledgerlens.synthetic import OUT_DIR

SYNTHETIC_CSV = OUT_DIR / "transactions.csv"


@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "test.db")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA_PATH.read_text())
    c.execute("PRAGMA foreign_keys = ON")
    yield c
    c.close()


# --- parsing -----------------------------------------------------------------

def test_synthetic_adapter_is_detected():
    assert detect_adapter(SYNTHETIC_CSV) is SyntheticCSV


def test_parse_returns_rows_with_sign_convention():
    rows = parse(SYNTHETIC_CSV)
    assert len(rows) > 500
    # §4 convention: negative = outflow, except income/refund
    for r in rows:
        if r.txn_type in ("income", "refund"):
            assert r.amount > 0, r
        else:
            assert r.amount < 0, r


def test_unknown_format_refuses_rather_than_guessing(tmp_path):
    junk = tmp_path / "mystery.csv"
    junk.write_text("col_a,col_b\n1,2\n")
    with pytest.raises(UnknownFormat):
        parse(junk)


def test_scanned_pdf_raises_instead_of_returning_empty(tmp_path):
    """A silent empty parse is indistinguishable from 'statement had no rows'."""
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF\n"
    )
    with pytest.raises(NoTextLayer):
        parse(pdf)


# --- dedup / idempotency -----------------------------------------------------

def test_content_hash_is_stable_and_field_sensitive():
    a = content_hash(1, "2026-01-05", -12.30, "SQ *BLUE BOTTLE AUSTIN TX")
    assert a == content_hash(1, "2026-01-05", -12.30, "SQ *BLUE BOTTLE AUSTIN TX")
    assert a != content_hash(1, "2026-01-05", -12.31, "SQ *BLUE BOTTLE AUSTIN TX")
    assert a != content_hash(2, "2026-01-05", -12.30, "SQ *BLUE BOTTLE AUSTIN TX")


def test_ingest_loads_rows(conn):
    result = ingest_file(conn, SYNTHETIC_CSV)
    assert result.adapter == "synthetic"
    assert result.inserted == result.parsed
    assert result.duplicates == 0
    (count,) = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
    assert count == result.inserted


def test_reingest_is_a_noop(conn):
    """§5.2 — overlapping statement ranges must cost nothing."""
    first = ingest_file(conn, SYNTHETIC_CSV)
    second = ingest_file(conn, SYNTHETIC_CSV)

    assert second.parsed == first.parsed
    assert second.inserted == 0
    assert second.duplicates == first.parsed

    (count,) = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
    assert count == first.inserted
