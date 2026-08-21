"""Tiered categorization — §5.4.

  1. corrections table             → 'user'     (you already fixed this merchant)
  2. merchants.default_category_id → 'learned'
  3. type + category_rules regex   → 'rule'
  4. LLM with the category list    → 'llm', with confidence stored

Anything below CONFIDENCE_THRESHOLD lands in a review queue rather than being
silently guessed — a wrong category is invisible in a total until someone asks
why groceries look high.

Tier 2 is the learning loop, and it only works because tier 4 writes back:
after the LLM categorizes a merchant once, `merchants.default_category_id` is
set and every later transaction for that merchant resolves free. Same shape as
the alias write-back in merchants.py.
"""

from __future__ import annotations

import enum
import functools
import re
import sqlite3
from datetime import datetime, timezone
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field

from ..seed import CATEGORIES

CONFIDENCE_THRESHOLD = 0.7
FALLBACK_CATEGORY = "uncategorized"

Tier = Literal["rule", "learned", "llm", "user"]

# Types whose category is implied by the type itself. No inference needed, and
# the rules tier should never get the chance to disagree.
TYPE_CATEGORY = {
    "income": "income",
    "transfer": "transfer",
    "fee": "fees",
}


class Categorization(NamedTuple):
    category_id: int | None
    category_name: str
    categorized_by: Tier
    confidence: float | None
    needs_review: bool
    tier: int  # 1=user 2=learned 3=rule 4=llm


# --- tier 4 schema -----------------------------------------------------------

# An enum, not a free string: the model cannot invent "food" or "misc", so the
# category_id lookup downstream cannot fail on an unknown name.
CategoryEnum = enum.Enum(
    "CategoryEnum", {name.upper(): name for name, _ in CATEGORIES}, type=str
)


class CategoryChoice(BaseModel):
    category: CategoryEnum = Field(description="Best-fitting category")
    confidence: float = Field(ge=0.0, le=1.0)


PROMPT = """Categorize this bank transaction into exactly one category.

Merchant:   {merchant}
Descriptor: {descriptor}

Guidance:
- Pharmacies (CVS, Walgreens) are health, not shopping.
- Gyms and fitness subscriptions are health, not subscriptions.
- Streaming and software subscriptions are subscriptions.
- Ride-share and fuel are transport; airlines and hotels are travel.
- Use "uncategorized" only when genuinely unclear, and lower the confidence.

Give an honest confidence: anything below {threshold} goes to human review,
which is the correct outcome for a genuine coin-flip."""


# --- rule loading ------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _compiled_rules(rules_key: tuple) -> list[tuple[re.Pattern, int]]:
    return [(re.compile(pattern, re.IGNORECASE), cid) for pattern, cid in rules_key]


def _rules(conn: sqlite3.Connection) -> list[tuple[re.Pattern, int]]:
    """SQLite has no REGEXP by default, so matching happens in Python."""
    rows = tuple(
        conn.execute(
            "SELECT pattern, category_id FROM category_rules ORDER BY priority, id"
        ).fetchall()
    )
    return _compiled_rules(tuple((r[0], r[1]) for r in rows))


def _category_ids(conn: sqlite3.Connection) -> dict[str, int]:
    return {name: cid for cid, name in conn.execute("SELECT id, name FROM categories")}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- dispatch ----------------------------------------------------------------

def categorize(
    conn: sqlite3.Connection,
    raw_descriptor: str,
    merchant_id: int | None,
    txn_type: str = "purchase",
    use_llm: bool = True,
) -> Categorization:
    """Walk the tiers in order; first hit wins."""
    ids = _category_ids(conn)
    names = {cid: name for name, cid in ids.items()}

    def result(cid: int, by: Tier, conf: float | None, tier: int) -> Categorization:
        return Categorization(
            category_id=cid,
            category_name=names.get(cid, FALLBACK_CATEGORY),
            categorized_by=by,
            confidence=conf,
            needs_review=conf is not None and conf < CONFIDENCE_THRESHOLD,
            tier=tier,
        )

    # tier 1 — an explicit user correction always wins
    row = conn.execute(
        """SELECT category_id FROM corrections
           WHERE (merchant_id IS NOT NULL AND merchant_id = ?)
              OR raw_descriptor = ?
           ORDER BY created_at DESC LIMIT 1""",
        (merchant_id, raw_descriptor),
    ).fetchone()
    if row:
        return result(row[0], "user", 1.0, tier=1)

    # tier 2 — learned default for this merchant
    if merchant_id is not None:
        row = conn.execute(
            "SELECT default_category_id FROM merchants WHERE id = ? AND default_category_id IS NOT NULL",
            (merchant_id,),
        ).fetchone()
        if row:
            return result(row[0], "learned", 0.95, tier=2)

    # tier 3a — the transaction type settles it outright
    if txn_type in TYPE_CATEGORY:
        return result(ids[TYPE_CATEGORY[txn_type]], "rule", 1.0, tier=3)

    # tier 3b — regex rules, highest priority first
    for pattern, cid in _rules(conn):
        if pattern.search(raw_descriptor):
            _learn(conn, merchant_id, cid)
            return result(cid, "rule", 0.9, tier=3)

    # tier 4 — LLM, then write back so this merchant is free next time
    if not use_llm:
        return result(ids[FALLBACK_CATEGORY], "rule", 0.0, tier=3)

    merchant_name = _merchant_name(conn, merchant_id) or raw_descriptor
    choice = _llm_categorize(merchant_name, raw_descriptor)
    cid = ids.get(choice.category.value, ids[FALLBACK_CATEGORY])

    if choice.confidence >= CONFIDENCE_THRESHOLD:
        _learn(conn, merchant_id, cid)

    return result(cid, "llm", choice.confidence, tier=4)


def _merchant_name(conn: sqlite3.Connection, merchant_id: int | None) -> str | None:
    if merchant_id is None:
        return None
    row = conn.execute("SELECT canonical_name FROM merchants WHERE id = ?", (merchant_id,)).fetchone()
    return row[0] if row else None


def _learn(conn: sqlite3.Connection, merchant_id: int | None, category_id: int) -> None:
    """Promote a resolution to the merchant default, so tier 2 catches it next time."""
    if merchant_id is None:
        return
    conn.execute(
        """UPDATE merchants SET default_category_id = ?, last_seen = ?
           WHERE id = ? AND default_category_id IS NULL""",
        (category_id, _now(), merchant_id),
    )


def _llm_categorize(merchant: str, descriptor: str) -> CategoryChoice:
    from ..llm import invoke_structured

    return invoke_structured(
        CategoryChoice,
        PROMPT.format(merchant=merchant, descriptor=descriptor, threshold=CONFIDENCE_THRESHOLD),
    )


def record_correction(
    conn: sqlite3.Connection,
    category_id: int,
    merchant_id: int | None = None,
    raw_descriptor: str | None = None,
) -> None:
    """§8 — the only path by which a user correction enters the system.

    Also clears the learned default so the correction actually takes effect;
    leaving it would let tier 2 keep answering with the old value.
    """
    conn.execute(
        """INSERT INTO corrections (merchant_id, raw_descriptor, category_id, created_at)
           VALUES (?, ?, ?, ?)""",
        (merchant_id, raw_descriptor, category_id, _now()),
    )
    if merchant_id is not None:
        conn.execute(
            "UPDATE merchants SET default_category_id = ? WHERE id = ?",
            (category_id, merchant_id),
        )
    conn.commit()
