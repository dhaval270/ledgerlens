"""Tiered merchant normalization — §5.3.

  1. alias lookup   — exact match on merchant_aliases.raw_pattern. Free.
  2. rule strip     — drop processor prefixes, store numbers, city/state tails,
                      embedded dates, refs. Re-check the alias table, then match
                      against known canonical names.
  3. LLM fallback   — only if 1 and 2 miss.
  4. write back     — every LLM resolution is inserted into merchant_aliases, so
                      the same descriptor never costs a second call.

The write-back is what makes this cheap *and* reproducible: descriptors repeat
heavily across statements, so the LLM tier decays toward zero as the alias table
fills. §9.3's "% resolved without an LLM call" measures exactly that.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field

Tier = Literal["rule", "llm", "user"]


class Resolution(NamedTuple):
    merchant_id: int
    canonical_name: str
    resolved_by: Tier
    confidence: float | None
    tier: int  # 1=alias 2=rule 3=llm — for the hit-rate metric


# --- tier 2: deterministic noise removal -------------------------------------

US_STATES = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
    "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"
).split()

# Order matters: prefixes come off before refs, refs before trailing geography.
STRIP_PATTERNS: list[tuple[str, str]] = [
    # payment processor prefixes
    (r"^(SQ|TST|SP|PY|PAYPAL|AMZN Mktp US|IC|EB)\s*\*\s*", ""),
    (r"^(WWW\.|HTTP://|HTTPS://)", ""),
    # "AMAZON.COM*G239CPJJB" / "TARGET.COM * 934U9C19Q" — merchant then a ref
    (r"\.COM\s*\*\s*[A-Z0-9]{6,}", ".COM"),
    (r"\s*\*\s*[A-Z0-9]{8,}\b", " "),
    # billing domains and phone numbers
    (r"\b[A-Z]+\.(COM|NET|ORG)/[A-Z]+\b", " "),
    (r"\bHELP\.[A-Z]+\.COM\b", " "),
    (r"\b\d{3}-\d{3}-\d{4}\b", " "),
    # store identifiers
    (r"#\s*\d+", " "),
    (r"\bT-\d+\b", " "),
    (r"\bP\d{6,}\b", " "),
    (r"\b\d{5,}\b", " "),
    # bare transaction refs: 8+ chars mixing letters and digits, no separator.
    # Merchant names essentially never look like this; auth codes always do.
    (r"\b(?=[A-Z0-9]*[0-9])(?=[A-Z0-9]*[A-Z])[A-Z0-9]{8,}\b", " "),
    # noise tokens
    (r"\bQPS\b|\bPPD ID:[A-Z0-9]*|\bONLINE\b|\bINC\.?\b|\bLLC\b", " "),
]

# Descriptors often append a city with no state to anchor it ("WHOLEFDS AUSTIN").
# A heuristic list, not exhaustive — it converts repeat LLM calls for the same
# merchant-in-a-different-city into free alias hits.
COMMON_CITIES = {
    "AUSTIN", "BOSTON", "SEATTLE", "DENVER", "CHICAGO", "PORTLAND", "ATLANTA",
    "DALLAS", "HOUSTON", "PHOENIX", "MIAMI", "ORLANDO", "DETROIT", "MINNEAPOLIS",
    "NASHVILLE", "PHILADELPHIA", "PITTSBURGH", "SACRAMENTO", "OAKLAND", "BROOKLYN",
    "MANHATTAN", "BALTIMORE", "CHARLOTTE", "RALEIGH", "COLUMBUS", "INDIANAPOLIS",
    "MILWAUKEE", "KANSAS CITY", "ST LOUIS", "LAS VEGAS", "SAN DIEGO", "SAN JOSE",
    "SAN FRANCISCO", "LOS ANGELES", "NEW YORK", "SALT LAKE", "TAMPA", "CLEVELAND",
}

TRAILING_DATE = re.compile(r"\s\d{4}$")
TRAILING_STATE = re.compile(r"\s(" + "|".join(US_STATES) + r")$")


def _strip_trailing_city(s: str) -> str:
    """Drop a trailing city name, but never the whole string.

    Two-word cities are checked first so "KANSAS CITY" doesn't leave "KANSAS".
    """
    words = s.split()
    for size in (2, 1):
        if len(words) <= size:
            continue
        candidate = " ".join(words[-size:])
        if candidate in COMMON_CITIES:
            return " ".join(words[:-size])
    return s


def strip_noise(raw_descriptor: str) -> str:
    """Tier 2 — reduce a descriptor toward its merchant token."""
    s = raw_descriptor.upper()

    for pattern, repl in STRIP_PATTERNS:
        s = re.sub(pattern, repl, s)

    s = re.sub(r"[^A-Z0-9&'\s\.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Trailing "<CITY> <STATE>", then a bare embedded MMDD, applied repeatedly
    # because descriptors stack them ("... AUSTIN TX 0725").
    for _ in range(3):
        before = s
        s = TRAILING_DATE.sub("", s).strip()
        if TRAILING_STATE.search(s):
            s = TRAILING_STATE.sub("", s).strip()
            s = re.sub(r"\s+\S+$", "", s).strip()  # drop the city too
        s = _strip_trailing_city(s)
        if s == before:
            break

    return s.strip(" -.")


def _key(text: str) -> str:
    """Comparison key: letters and digits only."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


# --- tier 3: LLM -------------------------------------------------------------

PROMPT = """You normalize bank statement descriptors to canonical business names.

Merchants already known to this ledger:
{known}

Descriptor: {descriptor}

Rules:
- If the descriptor is the same business as one listed above, return that string
  EXACTLY. Splitting one business across two names corrupts every total.
- Otherwise return the parent brand, never a store format or a location.
  "SHELL SERVICE STATION" -> "Shell". "MARRIOTT AUSTIN" -> "Marriott".
- Title case, common trading name: "Trader Joe's", "Blue Bottle Coffee".
- Non-businesses (fees, transfers) get a short generic label.

Return JSON: {{"canonical_name": "<name>", "confidence": <0.0-1.0>}}"""


class MerchantName(BaseModel):
    """Enforced output shape for tier 3.

    Plain JSON mode was not enough here: the model returned
    {"WHOLEFDS": {"canonical_name": ...}} for some inputs — valid JSON, wrong
    shape. Schema enforcement makes that unrepresentable.
    """

    canonical_name: str = Field(description="Parent brand in title case")
    confidence: float = Field(ge=0.0, le=1.0)


def _llm_resolve(descriptor: str, known: list[str]) -> tuple[str, float]:
    """Resolve one descriptor.

    `descriptor` must be the *stripped* form — passing the raw string leaks its
    noise into the canonical name ("Marriott Austin TX"), which then splits the
    merchant and silently halves every aggregate over it.
    """
    from ..llm import invoke_structured

    listing = "\n".join(f"- {n}" for n in sorted(known)) if known else "(none yet)"
    result = invoke_structured(
        MerchantName, PROMPT.format(descriptor=descriptor, known=listing)
    )
    name = result.canonical_name.strip()
    if not name:
        raise ValueError(f"LLM returned an empty canonical_name for {descriptor!r}")
    return name, result.confidence


# --- storage -----------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^A-Z0-9]+", name.upper()) if t]


def _is_variant(a: str, b: str) -> bool:
    """True when one name is a token-prefix extension of the other.

    "Shell" / "Shell Oil" -> variant.        "Spotify" / "Spotify USA" -> variant.
    "American Airlines" / "American Eagle" -> NOT a variant, and that restraint
    is the point: a false merge fuses two real businesses into one total, which
    is far harder to notice than a split. Prefix-only keeps false merges to
    cases where one name genuinely contains the other.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb or ta == tb:
        return ta == tb
    shorter, longer = (ta, tb) if len(ta) < len(tb) else (tb, ta)
    return longer[: len(shorter)] == shorter


def _upsert_merchant(conn: sqlite3.Connection, canonical: str) -> int:
    row = conn.execute(
        "SELECT id FROM merchants WHERE canonical_name = ?", (canonical,)
    ).fetchone()
    if row:
        return row[0]

    # Reuse an existing merchant when the new name is only a variant of it.
    for mid, existing in conn.execute("SELECT id, canonical_name FROM merchants"):
        if _is_variant(canonical, existing):
            return mid

    return conn.execute(
        "INSERT INTO merchants (canonical_name, first_seen, last_seen) VALUES (?, ?, ?)",
        (canonical, _now(), _now()),
    ).lastrowid


def _record_alias(
    conn: sqlite3.Connection, raw_pattern: str, merchant_id: int,
    resolved_by: Tier, confidence: float | None,
) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO merchant_aliases
           (raw_pattern, merchant_id, resolved_by, confidence, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (raw_pattern, merchant_id, resolved_by, confidence, _now()),
    )


# --- dispatch ----------------------------------------------------------------

# On a peer-to-peer transfer the merchant is the person, not the rail. Resolving
# "Zelle Payment To Manan Parikh Jpm99Cmxs17C" to `Zelle` collapses everyone you
# have ever paid into one merchant, so "how much did I send to Manan?" cannot be
# answered by a join and the counterparty survives only inside raw_descriptor.
# Observed live: the model wrote `WHERE m.canonical_name = 'Dhaval Patel'`, which
# is the right shape against a schema where payees are merchants — so make them
# merchants rather than teach the model to grep a text column.
_P2P = re.compile(
    r"^(?:zelle|venmo|cash\s*app|paypal)\s+(?:payment\s+)?(?:from|to)\s+(.+)$",
    re.IGNORECASE,
)
# Trailing confirmation codes: Jpm99Cmxs17C, Wfct129X5Tyy, 30099853170. A digit
# is required so a genuine surname is never mistaken for a reference.
_REFERENCE = re.compile(r"^(?=\S*\d)[A-Za-z0-9]{6,}$")


def counterparty(raw_descriptor: str) -> str | None:
    """The person on the other side of a P2P transfer, or None."""
    match = _P2P.match(raw_descriptor.strip())
    if not match:
        return None
    words = match.group(1).split()
    # Always keep one token. Zelle recipients are sometimes a phone number
    # rather than a name ("Zelle Payment To 1413409425 29488151212"), and
    # stripping every digit-bearing token leaves nothing — which collapsed that
    # payment back into a generic `Zelle` and lost the only identifier it had.
    while len(words) > 1 and _REFERENCE.match(words[-1]):
        words.pop()
    name = " ".join(words).title()
    return name or None


def resolve(conn: sqlite3.Connection, raw_descriptor: str, use_llm: bool = True) -> Resolution:
    """Walk the tiers in order, writing back any LLM resolution."""

    # tier 1 — exact alias on the untouched descriptor
    row = conn.execute(
        """SELECT m.id, m.canonical_name, a.resolved_by, a.confidence
           FROM merchant_aliases a JOIN merchants m ON m.id = a.merchant_id
           WHERE a.raw_pattern = ?""",
        (raw_descriptor,),
    ).fetchone()
    if row:
        return Resolution(row[0], row[1], row[2], row[3], tier=1)

    # tier 2a — peer-to-peer transfer: the counterparty is the merchant.
    # Deterministic, so it runs before the generic stripping that would otherwise
    # cluster every Zelle payment under one name.
    person = counterparty(raw_descriptor)
    if person:
        merchant_id = _upsert_merchant(conn, person)
        _record_alias(conn, raw_descriptor, merchant_id, "rule", 1.0)
        return Resolution(merchant_id, person, "rule", 1.0, tier=2)

    # tier 2 — strip, then re-check aliases and known canonical names
    stripped = strip_noise(raw_descriptor)
    if stripped:
        row = conn.execute(
            """SELECT m.id, m.canonical_name, a.confidence
               FROM merchant_aliases a JOIN merchants m ON m.id = a.merchant_id
               WHERE a.raw_pattern = ?""",
            (stripped,),
        ).fetchone()
        if row:
            _record_alias(conn, raw_descriptor, row[0], "rule", row[2])
            return Resolution(row[0], row[1], "rule", row[2], tier=2)

        key = _key(stripped)
        for mid, canonical in conn.execute("SELECT id, canonical_name FROM merchants"):
            ck = _key(canonical)
            if ck and (ck == key or key.startswith(ck) or ck.startswith(key)):
                _record_alias(conn, raw_descriptor, mid, "rule", 0.9)
                return Resolution(mid, canonical, "rule", 0.9, tier=2)

    # tier 3 — LLM, then write back so this descriptor is free next time
    if not use_llm:
        merchant_id = _upsert_merchant(conn, stripped or raw_descriptor)
        _record_alias(conn, raw_descriptor, merchant_id, "rule", 0.3)
        return Resolution(merchant_id, stripped or raw_descriptor, "rule", 0.3, tier=2)

    known = [n for (n,) in conn.execute("SELECT canonical_name FROM merchants")]
    canonical, confidence = _llm_resolve(stripped or raw_descriptor, known)
    merchant_id = _upsert_merchant(conn, canonical)
    _record_alias(conn, raw_descriptor, merchant_id, "llm", confidence)
    _record_alias(conn, stripped, merchant_id, "llm", confidence)  # generalize
    return Resolution(merchant_id, canonical, "llm", confidence, tier=3)
