"""LLM narration of the monthly digest — §7.

The model receives findings from checks.py and never raw transactions. It
narrates undismissed insights into a short digest; it does not compute anything.
"""

from __future__ import annotations

import sqlite3


def narrate(conn: sqlite3.Connection, period: str) -> str:
    """Turn undismissed insights for a YYYY-MM period into a short digest."""