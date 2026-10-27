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


# ---------------------------------------------------------------- Phase 1

import random  # noqa: E402

from tau2.domains.telecom.environment import get_tasks  # noqa: E402
from tau2.lifecycle.world import (  # noqa: E402
    TEMPLATE_NAME,
    append_events,
    branch_id,
    check_applicable,
    choose_task,
    instantiate_task,
    make_customers,
    project_state,
    register_customers,
)


def _state(**overrides):
    c = make_customers(1, seed=1)[0]
    return {
        **c.model_dump(exclude={"traits"}),
        "plan_id": "P1002",
        "account_status": "Active",
        **overrides,
    }


def test_append_events_replay_writes_nothing_twice(tmp_path):
    conn = open_ledger(tmp_path / "l.sqlite")
    ev = {"event_id": "x", "branch_id": "P1:v", "virtual_day": 1, "type": "csat"}
    append_events(conn, [ev, ev])
    append_events(conn, [ev])
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 1


def test_make_customers_is_deterministic_and_distinct():
    a, b = make_customers(20, seed=7), make_customers(20, seed=7)
    assert a == b
    assert len({c.pop_id for c in a}) == 20
    assert {c.persona for c in a} <= {"None", "Easy", "Hard"}
    assert make_customers(20, seed=8) != a


def test_projection_follows_events_and_keeps_traits_private(tmp_path):
    conn = open_ledger(tmp_path / "l.sqlite")
    customers = make_customers(2, seed=3)
    register_customers(conn, customers, variants=["v1", "v2"])
    b = branch_id(customers[0].pop_id, "v1")
    append_events(
        conn,
        [
            {
                "event_id": f"{b}:10:plan",
                "branch_id": b,
                "virtual_day": 10,
                "type": "plan_changed",
                "payload": {"plan_id": "P1001"},
            }
        ],
    )
    assert project_state(conn, b, 9)["plan_id"] == "P1002"
    assert project_state(conn, b, 10)["plan_id"] == "P1001"
    other = branch_id(customers[0].pop_id, "v2")
    assert project_state(conn, other, 10)["plan_id"] == "P1002"  # branches isolated
    public = " ".join(r[0] for r in conn.execute("SELECT payload FROM events"))
    assert "base_churn" not in public and "patience" not in public
    assert conn.execute("SELECT count(*) FROM oracle").fetchone()[0] == 4


def test_instantiated_task_is_about_the_customer():
    template = get_tasks("small")[0]
    state = _state()
    task = instantiate_task(template, state)
    dumped = task.model_dump_json()
    assert TEMPLATE_NAME not in dumped
    assert state["full_name"] in task.user_scenario.instructions.known_info
    c1001 = next(
        c
        for c in task.initial_state.initialization_data.agent_data["customers"]
        if c["customer_id"] == "C1001"
    )
    assert c1001["full_name"] == state["full_name"]


@pytest.mark.parametrize("split", ["small", "base"])
def test_retarget_equivalence(split):
    """D13: a fresh customer on the template's plan grades exactly like the original."""
    tasks = get_tasks(split)
    state = _state()
    passing = 0
    for t in tasks:
        original = check_applicable(t)
        assert check_applicable(instantiate_task(t, state)) == original, t.id
        passing += original is None
    assert passing >= 0.9 * len(tasks)  # upstream: 19/20 small, 108/114 base


def test_plan_change_reaches_the_environment():
    template = get_tasks("small")[0]
    task = instantiate_task(template, _state(plan_id="P1001"))
    line = next(
        ln
        for ln in task.initial_state.initialization_data.agent_data["lines"]
        if ln["line_id"] == "L1002"
    )
    assert line["plan_id"] == "P1001"


def test_choose_task_matches_persona_and_reports_swaps():
    state = _state(persona="Hard")
    task, swaps = choose_task(get_tasks("small"), state, random.Random(0))
    assert task.id.endswith("[PERSONA:Hard]")
    assert all("reason" in s for s in swaps)


def _line(task):
    return next(
        ln
        for ln in task.initial_state.initialization_data.agent_data["lines"]
        if ln["line_id"] == "L1002"
    )


def test_billing_cycle_resets_usage_at_30_days_not_on_plan_change(tmp_path):
    conn = open_ledger(tmp_path / "l.sqlite")
    customers = make_customers(1, seed=3)
    register_customers(conn, customers, variants=["v1"])
    b = branch_id(customers[0].pop_id, "v1")
    append_events(
        conn,
        [
            {
                "event_id": f"{b}:10:plan",
                "branch_id": b,
                "virtual_day": 10,
                "type": "plan_changed",
                "payload": {"plan_id": "P1001"},
            }
        ],
    )
    template = get_tasks("small")[0]
    before = _line(instantiate_task(template, project_state(conn, b, 29)))
    after = _line(instantiate_task(template, project_state(conn, b, 30)))
    assert before["plan_id"] == "P1001" and before["data_used_gb"] == 8.7
    assert after["data_used_gb"] == 0.0 and after["data_refueling_gb"] == 0.0


def test_goodwill_counter_accumulates_and_resets_yearly(tmp_path):
    conn = open_ledger(tmp_path / "l.sqlite")
    customers = make_customers(1, seed=3)
    register_customers(conn, customers, variants=["v1"])
    b = branch_id(customers[0].pop_id, "v1")
    append_events(
        conn,
        [
            {
                "event_id": f"{b}:{d}:credit",
                "branch_id": b,
                "virtual_day": d,
                "type": "goodwill_credit_applied",
                "payload": {"amount": 20.0},
            }
            for d in (10, 400)
        ],
    )
    assert project_state(conn, b, 5)["goodwill_credit_used_this_year"] == 25.0
    assert project_state(conn, b, 20)["goodwill_credit_used_this_year"] == 45.0
    assert project_state(conn, b, 380)["goodwill_credit_used_this_year"] == 0.0
    assert project_state(conn, b, 410)["goodwill_credit_used_this_year"] == 20.0
