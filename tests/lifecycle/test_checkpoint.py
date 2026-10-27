"""Phase 2: checkpoints, hazard, labels and KPIs.

The four OV7 end-to-end tests run a real telecom_lifecycle environment: the
call is built from the projected customer state, a scripted agent acts on the
env, the post-call DB is persisted through the ledger, and labels are read back.
"""

import pytest

from tau2.lifecycle.checkpoint import (
    HazardConfig,
    daily_tick,
    outcome_from_env,
    record_conversation,
)
from tau2.lifecycle.kpis import cohort_kpis, window_labels
from tau2.lifecycle.telecom_ext import get_environment, make_intent_task
from tau2.lifecycle.world import (
    branch_id,
    instantiate_task,
    make_customers,
    open_ledger,
    project_state,
    register_customers,
)

SEED = 11


@pytest.fixture
def world(tmp_path):
    conn = open_ledger(tmp_path / "ledger.sqlite")
    customer = make_customers(1, seed=5)[0]
    register_customers(conn, [customer], variants=["A", "B"])
    return conn, customer


def _call(conn, branch, day, intent, accepts, agent="gold", csat=4, infra=False):
    """Run one intent call on the lifecycle env with a scripted agent and record it."""
    state = project_state(conn, branch, day)
    task = instantiate_task(make_intent_task(state, intent, accepts), state)
    env = get_environment()
    init = task.initial_state
    env.set_state(init.initialization_data, init.initialization_actions, [])
    errors = 0
    for a in task.evaluation_criteria.actions if agent == "gold" else []:
        try:
            env.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception:
            errors += 1
    outcome = outcome_from_env(
        env,
        conversation_id=f"c{day}",
        task_id=task.id,
        intent=intent,
        accepts_offer=accepts,
        task_success=None if infra else float(agent == "gold"),
        tool_errors=errors,
        infra_error=infra,
    )
    return record_conversation(
        conn,
        branch,
        day,
        outcome,
        traits={"patience": 0.5},
        csat_fn=lambda o, t: (csat, "ok"),
    )


# ---------------------------------------------------------------- OV7 end-to-end


def test_e2e_downgrade(world):
    conn, c = world
    b = branch_id(c.pop_id, "A")
    res = _call(conn, b, 5, "downgrade", None)
    assert res["intent_outcome"] == "changed"
    assert project_state(conn, b, 5)["plan_id"] == "P1001"
    assert project_state(conn, branch_id(c.pop_id, "B"), 5)["plan_id"] == "P1002"
    (row,) = window_labels(conn, cutoff_day=40)
    assert row["churn_or_downgrade_28d"] == 1 and row["csat"] == 4


def test_e2e_accepted_retention_suppresses_cancel_for_30_days(world):
    conn, c = world
    b = branch_id(c.pop_id, "A")
    res = _call(conn, b, 5, "cancel", True)
    assert res["intent_outcome"] == "retained"
    state = project_state(conn, b, 6)
    assert not state["churned"] and state["goodwill_credit_used_this_year"] == 45.0
    angry = HazardConfig(
        w_price=50.0, organic_contacts_per_30d=0
    )  # leave intent is certain unless suppressed
    traits = {**c.traits, "price_sensitivity": 0.9}
    for day in range(6, 40):
        daily_tick(conn, b, day, traits, SEED, angry)
    contacts = conn.execute(
        "SELECT virtual_day FROM events WHERE branch_id=? AND type='contact_scheduled'",
        (b,),
    ).fetchall()
    # first leave intent lands on day 35, the first day after the 30-day suppression
    assert contacts and min(d for (d,) in contacts) == 35


def test_e2e_refused_retention_cancels_and_stops_the_customer(world):
    conn, c = world
    b = branch_id(c.pop_id, "A")
    res = _call(conn, b, 5, "cancel", False)
    assert res["intent_outcome"] == "cancelled"
    assert project_state(conn, b, 5)["churned"]
    assert daily_tick(conn, b, 6, c.traits, SEED) == []
    (row,) = window_labels(conn, cutoff_day=40)
    assert row["churn_or_downgrade_28d"] == 1


@pytest.mark.parametrize("infra", [False, True])
def test_e2e_failed_call_is_blocked_and_counts_as_churn(world, infra):
    conn, c = world
    b = branch_id(c.pop_id, "A")
    res = _call(conn, b, 5, "cancel", True, agent="none", csat=5, infra=infra)
    assert res["intent_outcome"] == "cancel_blocked"
    assert res["csat"] == 1  # forced, the rater's 5 is never used
    assert not project_state(conn, b, 5)["churned"]
    assert project_state(conn, b, 6)["churned"]  # self-serve cancel next day
    (row,) = window_labels(conn, cutoff_day=40)
    assert row["churn_or_downgrade_28d"] == 1  # obstruction is never a positive label


# ---------------------------------------------------------------- units


def test_csat_rater_failure_marks_pending_and_writes_no_score(world):
    conn, c = world
    b = branch_id(c.pop_id, "A")
    state = project_state(conn, b, 1)

    def broken(o, t):
        raise RuntimeError("provider down")

    from tau2.lifecycle.checkpoint import ConversationOutcome

    out = ConversationOutcome(
        conversation_id="c1",
        task_id="t",
        task_success=1.0,
        plan_id=state["plan_id"],
        line_status="Active",
        goodwill_used=25.0,
    )
    res = record_conversation(conn, b, 1, out, c.traits, broken)
    assert res["pending"] and res["csat"] is None
    types = [
        t for (t,) in conn.execute("SELECT type FROM events WHERE branch_id=?", (b,))
    ]
    assert "csat_pending" in types and "csat" not in types


def test_record_conversation_replay_is_a_no_op(world):
    conn, c = world
    b = branch_id(c.pop_id, "A")
    _call(conn, b, 5, "downgrade", None)
    n = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    m = conn.execute("SELECT count(*) FROM oracle").fetchone()[0]
    state = project_state(conn, b, 5)
    assert state["plan_id"] == "P1001"
    # replaying the same call id against the original pre-call state changes nothing
    conn.execute("SELECT 1")
    from tau2.lifecycle.checkpoint import ConversationOutcome

    again = ConversationOutcome(
        conversation_id="c5",
        task_id="x",
        intent="downgrade",
        task_success=1.0,
        plan_id="P1001",
        line_status="Active",
        goodwill_used=25.0,
    )
    record_conversation(conn, b, 5, again, c.traits, lambda o, t: (4, ""))
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == n
    assert conn.execute("SELECT count(*) FROM oracle").fetchone()[0] == m


def test_daily_draws_match_across_variants_until_histories_diverge(world):
    conn, c = world
    a, b = branch_id(c.pop_id, "A"), branch_id(c.pop_id, "B")
    for day in range(1, 30):
        ea = [
            (e["type"], e["payload"]) for e in daily_tick(conn, a, day, c.traits, SEED)
        ]
        eb = [
            (e["type"], e["payload"]) for e in daily_tick(conn, b, day, c.traits, SEED)
        ]
        assert ea == eb


def test_hazard_rises_after_a_bad_call(world):
    conn, c = world
    a, b = branch_id(c.pop_id, "A"), branch_id(c.pop_id, "B")
    _call(conn, a, 1, "downgrade", None, agent="none", csat=1)  # blocked, CSAT 1
    daily_tick(conn, a, 2, c.traits, SEED)
    daily_tick(conn, b, 2, c.traits, SEED)
    import json

    def p(branch):
        (payload,) = conn.execute(
            "SELECT payload FROM oracle WHERE branch_id=? AND kind='hazard' AND virtual_day=2",
            (branch,),
        ).fetchone()
        return json.loads(payload)["p_leave"]

    assert p(a) > p(b)


def test_labels_are_immature_until_the_window_closes(world):
    conn, c = world
    b = branch_id(c.pop_id, "A")
    _call(conn, b, 5, "upgrade", None)
    (row,) = window_labels(conn, cutoff_day=20)
    assert row["churn_or_downgrade_28d"] is None and row["repeat_contact_7d"] == 0
    (row,) = window_labels(conn, cutoff_day=33)
    assert row["churn_or_downgrade_28d"] == 0


def test_cohort_kpis_count_customers_and_keep_task_success_separate(world):
    conn, c = world
    _call(conn, branch_id(c.pop_id, "A"), 5, "cancel", False)
    _call(conn, branch_id(c.pop_id, "B"), 5, "cancel", True, agent="none", infra=True)
    k = cohort_kpis(conn, cutoff_day=40)
    assert k["A"]["churn_rate"] == 1.0 and k["A"]["task_success_rate"] == 1.0
    assert k["B"]["churn_rate"] == 1.0 and k["B"]["task_success_rate"] is None
    assert k["B"]["infra_errors"] == 1 and k["B"]["csat_mean"] == 1


def test_infra_error_on_a_support_call_gets_no_csat(world):
    from tau2.lifecycle.checkpoint import ConversationOutcome

    conn, c = world
    b = branch_id(c.pop_id, "A")
    out = ConversationOutcome(
        conversation_id="c1",
        task_id="t",
        infra_error=True,
        plan_id="P1002",
        line_status="Active",
        goodwill_used=25.0,
    )
    res = record_conversation(conn, b, 1, out, c.traits, lambda o, t: (5, ""))
    assert res["csat"] is None and not res["pending"]
    types = [
        t for (t,) in conn.execute("SELECT type FROM events WHERE branch_id=?", (b,))
    ]
    assert "csat" not in types and "csat_pending" not in types
