"""Lifecycle ledger: the single source of truth for customers across conversations.

`events` holds everything a real company could observe (conversations, CSAT,
lifecycle actions). `oracle` holds private simulator truth (hidden traits,
hazard terms, injected failures) and is never read by exports or the default UI.
A branch is one population customer under one agent variant.
"""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,
    branch_id   TEXT NOT NULL,
    virtual_day INTEGER NOT NULL,
    type        TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_branch_day ON events (branch_id, virtual_day);

CREATE TABLE IF NOT EXISTS oracle (
    oracle_id   TEXT PRIMARY KEY,
    branch_id   TEXT NOT NULL,
    virtual_day INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS oracle_branch_day ON oracle (branch_id, virtual_day);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def open_ledger(path: str | Path) -> sqlite3.Connection:
    """Open (or create) a ledger file in WAL mode with the current schema."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
    elif int(row[0]) != SCHEMA_VERSION:
        conn.close()
        raise ValueError(
            f"Ledger {path} has schema v{row[0]}, this code expects v{SCHEMA_VERSION}"
        )
    return conn
