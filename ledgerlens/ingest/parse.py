"""PDF/CSV → raw rows — §5.1.

Bank formats vary wildly, so this is an adapter registry, not a universal
parser. Each adapter declares whether it recognizes a file; dispatch picks the
one that matches and refuses when none does.

Refusing is the important part. A misparsed statement writes plausible-looking
wrong numbers into the ledger, and every downstream number inherits the error —
which is exactly the failure mode the verifier (§6.7) cannot catch, because by
then the bad data *is* the ground truth. Wrong beats missing here, so parse
failures are loud.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, NamedTuple, Protocol

import pdfplumber


class RawRow(NamedTuple):
    posted_date: str  # ISO YYYY-MM-DD
    descriptor: str
    amount: float     # negative = outflow
    txn_type: str = "purchase"


# --- failure modes, each distinguishable by the caller -----------------------

class StatementError(Exception):
    """Base for anything that makes a file unparseable."""


class EncryptedStatement(StatementError):
    """Password-protected PDF. Most banks encrypt statement downloads."""


class NoTextLayer(StatementError):
    """Scanned image with no extractable text.

    There is no OCR in the §2 stack, so this is genuinely unsupported. It must
    raise rather than return [] — a silent empty parse reads as "statement had
    no transactions", which is indistinguishable from success.
    """


class UnknownFormat(StatementError):
    """No adapter recognized the file. Never guess an adapter."""


# --- adapters ----------------------------------------------------------------

class Adapter(Protocol):
    name: ClassVar[str]

    @classmethod
    def matches(cls, path: Path, probe: str) -> bool: ...

    @classmethod
    def parse(cls, path: Path) -> list[RawRow]: ...


@dataclass
class SyntheticCSV:
    """The generator's own output — §10. Lets the whole ingest chain be
    exercised end to end before any real bank adapter exists."""

    name: ClassVar[str] = "synthetic"
    REQUIRED: ClassVar[set[str]] = {"posted_date", "amount", "raw_descriptor", "type"}

    @classmethod
    def matches(cls, path: Path, probe: str) -> bool:
        if path.suffix.lower() != ".csv":
            return False
        header = probe.splitlines()[0] if probe else ""
        return cls.REQUIRED.issubset({c.strip() for c in header.split(",")})

    @classmethod
    def parse(cls, path: Path) -> list[RawRow]:
        with path.open(newline="") as fh:
            return [
                RawRow(
                    posted_date=r["posted_date"],
                    descriptor=r["raw_descriptor"],
                    amount=float(r["amount"]),
                    txn_type=r.get("type", "purchase"),
                )
                for r in csv.DictReader(fh)
            ]


REGISTRY: list[type[Adapter]] = [SyntheticCSV]


# --- dispatch ----------------------------------------------------------------

PROBE_BYTES = 4096


def _probe(path: Path) -> str:
    """First chunk of extractable text, used only for adapter matching.

    Raises before returning if the file is structurally unusable, so callers get
    a specific error instead of a generic no-match.
    """
    if path.suffix.lower() == ".pdf":
        try:
            with pdfplumber.open(path) as pdf:
                text = (pdf.pages[0].extract_text() or "") if pdf.pages else ""
        except Exception as exc:  # pdfminer raises several distinct types here
            if "password" in str(exc).lower() or "encrypt" in str(exc).lower():
                raise EncryptedStatement(f"{path.name} is password-protected") from exc
            raise StatementError(f"{path.name} could not be opened: {exc}") from exc

        if not text.strip():
            raise NoTextLayer(
                f"{path.name} has no text layer — likely a scan. OCR is not in the stack."
            )
        return text

    return path.read_text(errors="replace")[:PROBE_BYTES]


def detect_adapter(path: Path) -> type[Adapter]:
    probe = _probe(path)
    for adapter in REGISTRY:
        if adapter.matches(path, probe):
            return adapter
    raise UnknownFormat(
        f"No adapter recognized {path.name}. "
        f"Known formats: {', '.join(a.name for a in REGISTRY)}."
    )


def parse(path: Path) -> list[RawRow]:
    """Dispatch to the matching adapter and return normalized raw rows."""
    return detect_adapter(path).parse(path)
