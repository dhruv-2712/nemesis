"""SQLite connection handling for NEMESIS.

Stdlib sqlite3 — no ORM. The persisted data (detection runs, reasoning logs) is
small and append-only, so a thin connection helper is enough and keeps the
deploy dependency-free. The DB path is configurable via NEMESIS_DB_PATH so tests
can point at a temp/in-memory file.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("backend/db/nemesis.db")


def db_path() -> Path:
    return Path(os.getenv("NEMESIS_DB_PATH", str(DEFAULT_DB_PATH)))


def get_connection() -> sqlite3.Connection:
    """Open a connection with row access by column name and FKs enabled."""
    path = db_path()
    if path.name != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
