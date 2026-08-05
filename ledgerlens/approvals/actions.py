"""The four approvable writes — §8.

  recategorize      merchant's category, retroactively, via `corrections`
  merge_merchants   fold one merchant's aliases and transactions into another
  set_budget        create or adjust a category budget
  dismiss_insight   stop surfacing one finding

Every action is split in two. `preview` reads current state and returns a
Proposal describing exactly what would change; `apply` performs the write. The
agent is only ever allowed to call the first half, which is what makes "the LLM
cannot mutate the ledger" a property of the code rather than a promise in a
prompt.

**A proposal is approval for one specific diff, not standing permission.** Each
carries a fingerprint of the `before` state, and `apply` recomputes it against
the live database and refuses on mismatch. Without that, an ingest landing
between "here is the diff" and "approve" would apply the write to a ledger the
user never saw — the approval UI would have shown a truthful diff and the
system would still have done something else. Cheap to add, and it is the whole
difference between showing a diff and honouring it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


class ApprovalError(Exception):
    """The proposal cannot be built or applied — bad subject, unknown action."""


class StaleProposal(ApprovalError):
    """The ledger changed after the diff was shown. Re-propose, do not guess."""


@dataclass
class Proposal:
    action: str
    summary: str            # one line, for the approval prompt
    before: dict
    after: dict
    params: dict
    affected_rows: int      # how many transactions this touches
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = _fingerprint(self.before)

    def as_dict(self) -> dict:
        return asdict(self)


def _fingerprint(before: dict) -> str:
    return hashlib.sha256(
        json.dumps(before, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _category_id(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()
    if not row:
        raise ApprovalError(f"unknown category: {name!r}")
    return row[0]


def _merchant(conn: sqlite3.Connection, merchant_id: int) -> sqlite3.Row:
    row = conn.execute(
        """SELECT m.id, m.canonical_name, m.default_category_id, c.name AS category
           FROM merchants m LEFT JOIN categories c ON c.id = m.default_category_id
           WHERE m.id = ?""",
        (merchant_id,),
    ).fetchone()
    if not row:
        raise ApprovalError(f"unknown merchant id: {merchant_id}")
    return row


# --- recategorize ------------------------------------------------------------

def preview_recategorize(conn: sqlite3.Connection, merchant_id: int, category: str) -> Proposal:
    merchant = _merchant(conn, merchant_id)
    target = _category_id(conn, category)
    if merchant["default_category_id"] == target:
        raise ApprovalError(f"{merchant['canonical_name']} is already {category}")

    (affected,) = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE merchant_id = ? AND IFNULL(category_id,-1) != ?",
        (merchant_id, target),
    ).fetchone()

    return Proposal(
        action="recategorize",
        summary=(f"{merchant['canonical_name']}: {merchant['category'] or 'uncategorized'} "
                 f"→ {category} ({affected} transactions)"),
        before={"merchant": merchant["canonical_name"],
                "category": merchant["category"],
                "transactions": affected},
        after={"merchant": merchant["canonical_name"], "category": category},
        params={"merchant_id": merchant_id, "category": category},
        affected_rows=affected,
    )


def apply_recategorize(conn: sqlite3.Connection, merchant_id: int, category: str) -> dict:
    """Record the correction *and* restate history.

    Writing only to `corrections` would fix future ingests and leave every past
    total wrong — the user corrects a merchant and last month's report still
    disagrees with them. The correction is the memory; the UPDATE is the point.
    """
    target = _category_id(conn, category)
    conn.execute(
        """INSERT INTO corrections (merchant_id, raw_descriptor, category_id, created_at)
           VALUES (?, NULL, ?, ?)""",
        (merchant_id, target, _now()),
    )
    conn.execute(
        "UPDATE merchants SET default_category_id = ? WHERE id = ?", (target, merchant_id)
    )
    cur = conn.execute(
        """UPDATE transactions
              SET category_id = ?, categorized_by = 'user', confidence = 1.0
            WHERE merchant_id = ? AND IFNULL(category_id,-1) != ?""",
        (target, merchant_id, target),
    )
    return {"transactions_updated": cur.rowcount, "category": category}


# --- merge merchants ---------------------------------------------------------

def preview_merge_merchants(conn: sqlite3.Connection, source_id: int, target_id: int) -> Proposal:
    if source_id == target_id:
        raise ApprovalError("cannot merge a merchant into itself")
    source, target = _merchant(conn, source_id), _merchant(conn, target_id)

    def counts(merchant_id: int) -> tuple[int, int]:
        (txns,) = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE merchant_id = ?", (merchant_id,)
        ).fetchone()
        (aliases,) = conn.execute(
            "SELECT COUNT(*) FROM merchant_aliases WHERE merchant_id = ?", (merchant_id,)
        ).fetchone()
        return txns, aliases

    src_txns, src_aliases = counts(source_id)
    tgt_txns, tgt_aliases = counts(target_id)

    return Proposal(
        action="merge_merchants",
        summary=(f"merge {source['canonical_name']} ({src_txns} txns) into "
                 f"{target['canonical_name']} ({tgt_txns} txns)"),
        before={"source": source["canonical_name"], "target": target["canonical_name"],
                "source_transactions": src_txns, "source_aliases": src_aliases,
                "target_transactions": tgt_txns, "target_aliases": tgt_aliases},
        after={"merchant": target["canonical_name"],
               "transactions": src_txns + tgt_txns,
               "aliases": src_aliases + tgt_aliases,
               "removed": source["canonical_name"]},
        params={"source_id": source_id, "target_id": target_id},
        affected_rows=src_txns,
    )


def apply_merge_merchants(conn: sqlite3.Connection, source_id: int, target_id: int) -> dict:
    source = _merchant(conn, source_id)
    conn.execute("UPDATE merchant_aliases SET merchant_id = ? WHERE merchant_id = ?",
                 (target_id, source_id))
    conn.execute("UPDATE recurring_series SET merchant_id = ? WHERE merchant_id = ?",
                 (target_id, source_id))
    cur = conn.execute("UPDATE transactions SET merchant_id = ? WHERE merchant_id = ?",
                       (target_id, source_id))
    # The surviving merchant inherits a category only if it lacks one; an
    # explicit category on the target is a decision and must not be overwritten.
    conn.execute(
        """UPDATE merchants SET default_category_id = COALESCE(default_category_id, ?)
           WHERE id = ?""",
        (source["default_category_id"], target_id),
    )
    conn.execute(
        """UPDATE merchants SET
             first_seen = MIN(IFNULL(first_seen,'9999'), (SELECT MIN(posted_date)
                          FROM transactions WHERE merchant_id = ?)),
             last_seen  = MAX(IFNULL(last_seen,''),      (SELECT MAX(posted_date)
                          FROM transactions WHERE merchant_id = ?))
           WHERE id = ?""",
        (target_id, target_id, target_id),
    )
    conn.execute("DELETE FROM merchants WHERE id = ?", (source_id,))
    return {"transactions_moved": cur.rowcount, "merged_into": target_id}


# --- budgets -----------------------------------------------------------------

def preview_set_budget(conn: sqlite3.Connection, category: str, limit_amount: float,
                       active_from: str | None = None) -> Proposal:
    if limit_amount <= 0:
        raise ApprovalError("a budget limit must be positive")
    category_id = _category_id(conn, category)
    existing = conn.execute(
        "SELECT id, limit_amount FROM budgets WHERE category_id = ? ORDER BY active_from DESC LIMIT 1",
        (category_id,),
    ).fetchone()

    current = existing["limit_amount"] if existing else None
    if current is not None and abs(current - limit_amount) < 0.005:
        raise ApprovalError(f"{category} budget is already {limit_amount:.2f}")

    return Proposal(
        action="set_budget",
        summary=(f"{category} budget: "
                 f"{'none' if current is None else f'{current:.2f}'} → {limit_amount:.2f}"),
        before={"category": category, "limit": current},
        after={"category": category, "limit": limit_amount},
        params={"category": category, "limit_amount": limit_amount,
                "active_from": active_from},
        affected_rows=0,
    )


def apply_set_budget(conn: sqlite3.Connection, category: str, limit_amount: float,
                     active_from: str | None = None) -> dict:
    category_id = _category_id(conn, category)
    active_from = active_from or _now()[:10]
    existing = conn.execute(
        "SELECT id FROM budgets WHERE category_id = ? ORDER BY active_from DESC LIMIT 1",
        (category_id,),
    ).fetchone()

    if existing:
        conn.execute("UPDATE budgets SET limit_amount = ?, active_from = ? WHERE id = ?",
                     (limit_amount, active_from, existing["id"]))
        budget_id = existing["id"]
    else:
        budget_id = conn.execute(
            """INSERT INTO budgets (category_id, period, limit_amount, active_from)
               VALUES (?, 'monthly', ?, ?)""",
            (category_id, limit_amount, active_from),
        ).lastrowid
    return {"budget_id": budget_id, "limit": limit_amount}


# --- dismiss insight ---------------------------------------------------------

def preview_dismiss_insight(conn: sqlite3.Connection, insight_id: int) -> Proposal:
    row = conn.execute(
        "SELECT id, kind, period, payload, dismissed FROM insights WHERE id = ?",
        (insight_id,),
    ).fetchone()
    if not row:
        raise ApprovalError(f"unknown insight id: {insight_id}")
    if row["dismissed"]:
        raise ApprovalError(f"insight {insight_id} is already dismissed")

    payload = json.loads(row["payload"])
    return Proposal(
        action="dismiss_insight",
        summary=f"dismiss {row['kind']} ({row['period']}): "
                f"{payload.get('merchant') or payload.get('category') or row['kind']}",
        before={"id": row["id"], "kind": row["kind"], "period": row["period"],
                "dismissed": 0},
        after={"id": row["id"], "dismissed": 1},
        params={"insight_id": insight_id},
        affected_rows=0,
    )


def apply_dismiss_insight(conn: sqlite3.Connection, insight_id: int) -> dict:
    cur = conn.execute("UPDATE insights SET dismissed = 1 WHERE id = ?", (insight_id,))
    return {"insight_id": insight_id, "dismissed": cur.rowcount}


# --- registry ----------------------------------------------------------------

ACTIONS = {
    "recategorize": (preview_recategorize, apply_recategorize),
    "merge_merchants": (preview_merge_merchants, apply_merge_merchants),
    "set_budget": (preview_set_budget, apply_set_budget),
    "dismiss_insight": (preview_dismiss_insight, apply_dismiss_insight),
}


def build_proposal(conn: sqlite3.Connection, action: str, params: dict) -> Proposal:
    """Read-only: describes the change without making it."""
    if action not in ACTIONS:
        raise ApprovalError(f"unknown action: {action!r} (expected one of {sorted(ACTIONS)})")
    preview, _ = ACTIONS[action]
    try:
        return preview(conn, **params)
    except TypeError as exc:
        raise ApprovalError(f"bad parameters for {action}: {exc}") from exc


def commit_proposal(conn: sqlite3.Connection, proposal: dict) -> dict:
    """Apply an approved proposal, refusing if the ledger moved underneath it."""
    action = proposal["action"]
    if action not in ACTIONS:
        raise ApprovalError(f"unknown action: {action!r}")

    fresh = build_proposal(conn, action, proposal["params"])
    if fresh.fingerprint != proposal["fingerprint"]:
        raise StaleProposal(
            f"the ledger changed since this diff was shown "
            f"({proposal['summary']!r}); re-propose to see the current one"
        )

    _, apply = ACTIONS[action]
    result = apply(conn, **proposal["params"])
    conn.commit()
    return result
