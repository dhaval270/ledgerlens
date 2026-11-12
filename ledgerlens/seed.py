"""Default taxonomy and rules — the deterministic tiers of §5.4 need something
to match against before any LLM is involved.

A caveat to keep honest in the README: these regexes were written knowing what
the synthetic catalog contains, so measuring them on synthetic data flatters
them. The rule-tier hit rate reported in §9.3 is an upper bound, not a forecast
for real statements.
"""

from __future__ import annotations

import sqlite3

# (name, is_essential)
CATEGORIES: list[tuple[str, bool]] = [
    ("groceries", True),
    ("rent", True),
    ("utilities", True),
    ("health", True),
    ("transport", False),
    ("dining", False),
    ("coffee", False),
    ("shopping", False),
    ("entertainment", False),
    ("subscriptions", False),
    ("travel", False),
    ("income", False),
    ("transfer", False),
    ("fees", False),
    ("uncategorized", False),
]

# (regex on raw_descriptor, category, priority) — lower priority wins first.
# Anchored on distinctive tokens rather than whole descriptors, since the noise
# around them varies per statement.
CATEGORY_RULES: list[tuple[str, str, int]] = [
    # unambiguous non-purchase flows
    (r"\bDIRECT DEP\b|\bPAYROLL\b|\bDIRECT DEPOSIT\b", "income", 10),
    (r"\bTRANSFER TO\b|\bTRANSFER FROM\b|\bONLINE TRANSFER\b", "transfer", 10),
    (r"\bMAINTENANCE FEE\b|\bATM FEE\b|\bOVERDRAFT\b|\bSERVICE CHARGE\b", "fees", 10),
    # housing / utilities
    (r"\bRENT\b|\bAPTS\b|\bAPARTMENTS\b|\bLEASING\b", "rent", 20),
    (r"\bCOMCAST\b|\bXFINITY\b|\bELECTRIC\b|\bWATER DEPT\b|\bGAS CO\b|\bVERIZON\b|\bAT&T\b",
     "utilities", 20),
    # recurring digital services
    (r"\bNETFLIX\b|\bSPOTIFY\b|\bHULU\b|\bADOBE\b|\bDISNEY\+?\b|\bAPPLE\.COM/BILL\b",
     "subscriptions", 30),
    # food
    (r"\bTRADER JOE|\bWHOLEFDS\b|\bWHOLE FOODS\b|\bSAFEWAY\b|\bKROGER\b|\bALDI\b|\bCOSTCO\b",
     "groceries", 40),
    (r"\bSTARBUCKS\b|\bBLUE BOTTLE\b|\bPEET'?S\b|\bDUNKIN\b|\bCOFFEE\b", "coffee", 45),
    (r"\bCHIPOTLE\b|\bMCDONALD|\bSUBWAY\b|\bPANERA\b|\bDOORDASH\b|\bUBER ?EATS\b|\bGRUBHUB\b",
     "dining", 50),
    # movement
    (r"\bUBER\b|\bLYFT\b|\bSHELL\b|\bCHEVRON\b|\bEXXON\b|\bMTA\b|\bPARKING\b", "transport", 55),
    (r"\bDELTA AIR\b|\bUNITED AIR\b|\bMARRIOTT\b|\bHILTON\b|\bAIRBNB\b|\bEXPEDIA\b",
     "travel", 55),
    # health before shopping — CVS/Walgreens are pharmacies first
    (r"\bCVS\b|\bWALGREENS\b|\bPHARMACY\b|\bPLANET FIT\b|\bPELOTON\b|\bDENTAL\b|\bCLINIC\b",
     "health", 60),
    (r"\bAMZN\b|\bAMAZON\b|\bTARGET\b|\bWALMART\b|\bBEST BUY\b|\bETSY\b", "shopping", 70),
    (r"\bAMC\b|\bCINEMARK\b|\bREGAL\b|\bTICKETMASTER\b|\bSTEAM GAMES\b", "entertainment", 70),
]


def seed(conn: sqlite3.Connection) -> dict:
    """Idempotent — safe to run on every startup."""
    conn.executemany(
        "INSERT OR IGNORE INTO categories (name, is_essential) VALUES (?, ?)",
        [(name, int(essential)) for name, essential in CATEGORIES],
    )

    ids = {name: cid for cid, name in conn.execute("SELECT id, name FROM categories")}

    existing = {row[0] for row in conn.execute("SELECT pattern FROM category_rules")}
    new_rules = [
        (pattern, ids[category], priority)
        for pattern, category, priority in CATEGORY_RULES
        if pattern not in existing
    ]
    conn.executemany(
        "INSERT INTO category_rules (pattern, category_id, priority) VALUES (?, ?, ?)",
        new_rules,
    )
    conn.commit()

    return {"categories": len(ids), "rules_added": len(new_rules)}
