"""Phase 3: campaigns run offline with a scripted agent (no LLM calls)."""

import json

import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.telecom.environment import get_tasks
from tau2.lifecycle.campaign import CampaignConfig, Variant, run_campaign
from tau2.lifecycle.checkpoint import HazardConfig, outcome_from_env
from tau2.lifecycle.kpis import cohort_kpis
from tau2.lifecycle.telecom_ext import INJECTED_ERRORS, get_environment, is_injected
from tau2.lifecycle.world import open_ledger

TEMPLATES = get_tasks("small")


def _cfg(**kw):
    base = dict(
        name="t",
        customers=4,
        days=21,
        variants=[
            Variant(name="good", agent_llm="x"),
            Variant(name="bad", agent_llm="x"),
        ],
        budget_usd=100.0,
        cost_bound_usd=0.01,
        max_concurrency=4,
        hazard=HazardConfig(organic_contacts_per_30d=6),
    )
    return CampaignConfig(**{**base, **kw})


def scripted_runner(calls=None, cancel_after=None, out_dir=None):
    """'good' runs the gold path, 'bad' does nothing. Optionally requests cancel."""

    def run(task, variant, conv_id, intent_json):
        env = get_environment(failure_rate=variant.failure_rate, salt="t")
        init = task.initial_state
        env.set_state(init.initialization_data, init.initialization_actions, [])
        good = variant.name == "good"
        errors = 0
        for a in task.evaluation_criteria.actions if good else []:
            try:
                env.make_tool_call(
                    tool_name=a.name, requestor=a.requestor, **a.arguments
                )
            except Exception:
                errors += 1
        if calls is not None:
            calls.append(conv_id)
            if cancel_after and len(calls) == cancel_after:
                (out_dir / "CANCEL").touch()
        return outcome_from_env(
            env,
            conversation_id=conv_id,
            task_id=task.id,
            task_success=float(good),
            tool_errors=errors,
            cost_usd=0.01,
            transcript=f"call {conv_id}",
            **json.loads(intent_json),
        )

    return run


def csat(o, t):
    return (5 if o.task_success else 2), ""


def _events(path):
    conn = open_ledger(path / "ledger.sqlite")
    return sorted(conn.execute("SELECT event_id, virtual_day, payload FROM events"))


def test_campaign_completes_and_separates_good_from_bad(tmp_path):
    status = run_campaign(_cfg(), tmp_path, scripted_runner(), csat, TEMPLATES)
    assert status["status"] == "complete"
    k = cohort_kpis(open_ledger(tmp_path / "ledger.sqlite"), cutoff_day=20)
    assert k["good"]["conversations"] > 0 and k["bad"]["conversations"] > 0
    assert (
        k["good"]["task_success_rate"] == 1.0 and k["bad"]["task_success_rate"] == 0.0
    )
    assert k["good"]["csat_mean"] > k["bad"]["csat_mean"]


def test_cancel_then_resume_matches_an_uninterrupted_run(tmp_path):
    full, split = tmp_path / "full", tmp_path / "split"
    run_campaign(_cfg(), full, scripted_runner(), csat, TEMPLATES)
    calls = []
    first = run_campaign(
        _cfg(),
        split,
        scripted_runner(calls, cancel_after=3, out_dir=split),
        csat,
        TEMPLATES,
    )
    assert first["status"] == "cancelled"
    (split / "CANCEL").unlink()
    second = run_campaign(_cfg(), split, scripted_runner(), csat, TEMPLATES)
    assert second["status"] == "complete"
    assert _events(full) == _events(split)


def test_resume_with_a_different_config_is_refused(tmp_path):
    run_campaign(_cfg(days=3), tmp_path, scripted_runner(), csat, TEMPLATES)
    with pytest.raises(ValueError, match="Start a new output directory"):
        run_campaign(_cfg(days=3, seed=8), tmp_path, scripted_runner(), csat, TEMPLATES)


def test_budget_stops_before_a_day_that_would_overrun(tmp_path):
    status = run_campaign(
        _cfg(budget_usd=0.05, cost_bound_usd=0.02),
        tmp_path,
        scripted_runner(),
        csat,
        TEMPLATES,
    )
    assert status["status"] == "budget_stopped"
    assert status["spent_usd"] <= 0.05


def test_first_contact_is_matched_across_variants(tmp_path):
    run_campaign(_cfg(), tmp_path, scripted_runner(), csat, TEMPLATES)
    conn = open_ledger(tmp_path / "ledger.sqlite")
    first = {}
    for branch, day, payload in conn.execute(
        "SELECT branch_id, virtual_day, payload FROM events WHERE type='conversation' ORDER BY virtual_day, rowid"
    ):
        pop, variant = branch.split(":", 1)
        task_id = json.loads(payload)["task_id"].split("|", 1)[1]
        first.setdefault((pop, variant), (day, task_id))
    for pop in {p for p, _ in first}:
        if (pop, "good") in first and (pop, "bad") in first:
            assert first[(pop, "good")] == first[(pop, "bad")]


def test_pending_csat_stops_the_day_and_resolves_on_resume(tmp_path):
    def broken(o, t):
        raise RuntimeError("rater down")

    first = run_campaign(_cfg(), tmp_path, scripted_runner(), broken, TEMPLATES)
    assert first["status"] == "csat_pending"
    second = run_campaign(_cfg(), tmp_path, scripted_runner(), csat, TEMPLATES)
    assert second["status"] == "complete"
    conn = open_ledger(tmp_path / "ledger.sqlite")
    pending = {
        json.loads(p)["conversation_id"]
        for (p,) in conn.execute("SELECT payload FROM events WHERE type='csat_pending'")
    }
    rated = {
        json.loads(p)["conversation_id"]
        for (p,) in conn.execute("SELECT payload FROM events WHERE type='csat'")
    }
    assert pending and pending <= rated


def test_injected_failures_are_deterministic_and_change_nothing():
    rate, salt = 0.3, "s"
    ids = [f"call_{i}" for i in range(400)]
    hits = [i for i in ids if is_injected(salt, i, rate)]
    assert 0.2 < len(hits) / len(ids) < 0.4
    env = get_environment(failure_rate=rate, salt=salt)
    before = env.get_db_hash()
    call = ToolCall(
        id=hits[0],
        name="refuel_data",
        arguments={"customer_id": "C1001", "line_id": "L1002", "gb_amount": 1.0},
        requestor="assistant",
    )
    first, second = env.get_response(call), env.get_response(call)
    assert first.error and first == second
    assert any(e in first.content for e in INJECTED_ERRORS)
    used = {
        env.get_response(call.model_copy(update={"id": h})).content for h in hits[:60]
    }
    assert len(used) >= 4  # many messages, not one marker string
    assert env.get_db_hash() == before


def test_provider_outage_stops_the_day_without_recording_it(tmp_path):
    def outage(task, variant, conv_id, intent_json):
        env = get_environment()
        return outcome_from_env(
            env,
            conversation_id=conv_id,
            task_id=task.id,
            infra_error=True,
            transcript="infra error: no credits",
            **json.loads(intent_json),
        )

    status = run_campaign(_cfg(), tmp_path, outage, csat, TEMPLATES)
    assert status["status"] == "infra_stopped" and "no credits" in status["error"]
    conn = open_ledger(tmp_path / "ledger.sqlite")
    assert not conn.execute("SELECT 1 FROM events WHERE type='conversation'").fetchone()
    resumed = run_campaign(_cfg(), tmp_path, scripted_runner(), csat, TEMPLATES)
    assert resumed["status"] == "complete"
