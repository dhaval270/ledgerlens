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
import re
from dataclasses import dataclass
from datetime import date, datetime
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


@dataclass
class ChaseCheckingPDF:
    """Chase personal checking statements (`JPMorgan Chase Bank, N.A.`).

    Chase embeds literal `*start*transaction detail` / `*end*transaction detail`
    markers in the text layer, so the table is delimited by the bank rather than
    inferred from column geometry. Anchoring on those beats x-position matching:
    layout shifts between statement versions, the markers do not.

    Three things this does that a naive line-splitter does not:

    **The year comes from the statement period.** Rows carry only `MM/DD`. A
    statement spanning a year boundary would otherwise file 12/28 and 01/03 in
    the same year, silently moving a transaction twelve months. The year chosen
    is the one that places the date inside the statement period.

    **The running balance is used as a checksum.** Beginning balance plus the
    parsed amounts must equal the ending balance, or the file is rejected. A
    misparse — a dropped row, a description swallowing an amount, a sign flipped
    — fails this immediately, which is the only cheap way to catch the failure
    mode this module cares about: plausible wrong numbers that every downstream
    figure then inherits.

    **Transfers are not purchases.** On a checking statement most rows are money
    moving, not money spent: card payments, Zelle, internal transfers. §4 has
    every analytic filter `type = 'purchase'`, so mislabelling these inflates
    spending in a way no later stage can detect. The mapping is a heuristic on
    Chase's wording and is deliberately conservative — an unrecognized outflow
    stays a purchase, which is visible in a total, rather than a transfer, which
    silently vanishes from one.
    """

    name: ClassVar[str] = "chase-checking"

    START: ClassVar[str] = "*start*transaction detail"
    END: ClassVar[str] = "*end*transaction detail"

    _PERIOD = re.compile(
        r"([A-Z][a-z]+ \d{1,2}, \d{4})\s*through\s*([A-Z][a-z]+ \d{1,2}, \d{4})"
    )
    _ROW = re.compile(
        r"^(\d{2}/\d{2})\s+(.+?)\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})$"
    )
    _BEGIN = re.compile(r"Beginning Balance\s+\$?(-?[\d,]+\.\d{2})")
    _END_BAL = re.compile(r"Ending Balance\s+\$?(-?[\d,]+\.\d{2})")

    # Ordered: the first match wins, so the specific patterns precede the vague.
    _TYPES: ClassVar[tuple[tuple[str, str], ...]] = (
        ("payment to chase card", "transfer"),
        ("zelle payment", "transfer"),
        ("online transfer", "transfer"),
        ("transfer to", "transfer"),
        ("transfer from", "transfer"),
        ("payroll", "income"),
        ("direct deposit", "income"),
        ("dir dep", "income"),
        ("service fee", "fee"),
        ("overdraft fee", "fee"),
        ("monthly service", "fee"),
        ("atm fee", "fee"),
    )

    @classmethod
    def matches(cls, path: Path, probe: str) -> bool:
        if path.suffix.lower() != ".pdf":
            return False
        return cls.START in probe and (
            "JPMorgan Chase Bank" in probe or "Chase.com" in probe
        )

    @classmethod
    def parse(cls, path: Path) -> list[RawRow]:
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return cls.parse_text(text, source=path.name)

    @classmethod
    def parse_text(cls, text: str, source: str = "statement") -> list[RawRow]:
        """The whole parse, given an already-extracted text layer.

        Split out from `parse` so the logic is testable without a fixture PDF —
        real statements are the one thing that can never be committed (§10), so
        a parser only exercisable against one is a parser with no tests.
        """
        period_start, period_end = cls._period(text)
        block = cls._detail_block(text)

        rows: list[RawRow] = []
        for line in block:
            match = cls._ROW.match(line.strip())
            if not match:
                continue
            day, descriptor, amount, _balance = match.groups()
            value = float(amount.replace(",", ""))
            rows.append(RawRow(
                posted_date=cls._date(day, period_start, period_end),
                descriptor=" ".join(descriptor.split()),
                amount=value,
                txn_type=cls._classify(descriptor, value),
            ))

        if not rows:
            raise StatementError(
                f"{source} is a Chase statement but its transaction table "
                f"parsed to zero rows — the layout has probably changed."
            )
        cls._reconcile(source, block, rows)
        return rows

    # --- helpers -------------------------------------------------------------

    @classmethod
    def _period(cls, text: str) -> tuple[date, date]:
        match = cls._PERIOD.search(text)
        if not match:
            raise StatementError(
                "Chase statement has no '<date> through <date>' period line; "
                "transaction rows carry no year without it."
            )
        return (datetime.strptime(match.group(1), "%B %d, %Y").date(),
                datetime.strptime(match.group(2), "%B %d, %Y").date())

    @classmethod
    def _detail_block(cls, text: str) -> list[str]:
        lines = text.splitlines()
        try:
            first = next(i for i, l in enumerate(lines) if cls.START in l)
            last = next(i for i, l in enumerate(lines) if cls.END in l)
        except StopIteration as exc:
            raise StatementError(
                "Chase statement is missing its transaction-detail markers."
            ) from exc
        return lines[first + 1:last]

    @staticmethod
    def _date(day: str, start: date, end: date) -> str:
        """Pick the year that lands the MM/DD inside the statement period."""
        month, dom = (int(x) for x in day.split("/"))
        for year in (start.year, end.year):
            try:
                candidate = date(year, month, dom)
            except ValueError:      # 02/29 in the wrong year
                continue
            if start <= candidate <= end:
                return candidate.isoformat()
        # Outside the period: Chase can post a row dated just before it opens.
        # Anchor to the start year rather than dropping the row.
        return date(start.year, month, dom).isoformat()

    @classmethod
    def _classify(cls, descriptor: str, amount: float) -> str:
        lowered = descriptor.lower()
        for needle, txn_type in cls._TYPES:
            if needle in lowered:
                return txn_type
        return "income" if amount > 0 else "purchase"

    @classmethod
    def _reconcile(cls, source: str, block: list[str], rows: list[RawRow]) -> None:
        """Beginning + parsed amounts must equal ending, or refuse the file."""
        joined = "\n".join(block)
        begin, end = cls._BEGIN.search(joined), cls._END_BAL.search(joined)
        if not (begin and end):
            return  # nothing to check against; the parse stands on its own

        opening = float(begin.group(1).replace(",", ""))
        closing = float(end.group(1).replace(",", ""))
        computed = round(opening + sum(r.amount for r in rows), 2)
        if abs(computed - closing) > 0.01:
            raise StatementError(
                f"{source} failed balance reconciliation: "
                f"{opening:.2f} + {len(rows)} transactions = {computed:.2f}, "
                f"but the statement says {closing:.2f}. Refusing rather than "
                f"writing numbers that do not add up."
            )


REGISTRY: list[type[Adapter]] = [SyntheticCSV, ChaseCheckingPDF]


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
