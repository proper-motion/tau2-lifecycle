import pytest

from tau2.lifecycle.world import SCHEMA_VERSION, open_ledger


def test_open_ledger_creates_schema(tmp_path):
    conn = open_ledger(tmp_path / "ledger.sqlite")

    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"events", "oracle", "meta"} <= tables
    indexes = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    assert {"events_branch_day", "oracle_branch_day"} <= indexes
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[
        0
    ] == str(SCHEMA_VERSION)


def test_reopen_is_idempotent_and_event_ids_are_unique(tmp_path):
    path = tmp_path / "ledger.sqlite"
    conn = open_ledger(path)
    conn.execute(
        "INSERT INTO events (event_id, branch_id, virtual_day, type) VALUES ('e1', 'P1:v1', 0, 'conversation')"
    )
    conn.commit()
    conn.close()

    conn = open_ledger(path)
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 1
    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO events (event_id, branch_id, virtual_day, type) VALUES ('e1', 'P1:v1', 0, 'conversation')"
        )


def test_schema_version_mismatch_is_refused(tmp_path):
    path = tmp_path / "ledger.sqlite"
    conn = open_ledger(path)
    conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="schema v999"):
        open_ledger(path)
