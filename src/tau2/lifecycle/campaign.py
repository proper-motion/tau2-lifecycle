"""Campaigns: run matched customer populations across agent variants in day waves.

Each virtual day: run every conversation due that day (all branches, in parallel),
record them through the single ledger writer, then tick every live branch.
Budget, cancel and the config hash are checked at day boundaries; resume
restarts from the last completed day and skips conversations already in the
ledger (ids are deterministic), so nothing recorded is paid for twice.

Deviation from D12: upstream `runner/batch.run_tasks` requires every task to
belong to a registered, language-pinned task set, and our tasks are generated
per customer. We use upstream's lowest layer (`run_simulation`, which still
grades natively by rebuilding the env from its registered domain name) under
our own thread pool.
"""

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

from tau2.data_model.tasks import Task
from tau2.lifecycle.checkpoint import (
    ConversationOutcome,
    CsatFn,
    HazardConfig,
    _rng,
    daily_tick,
    llm_csat_rater,
    outcome_from_env,
    record_conversation,
)
from tau2.lifecycle.telecom_ext import (
    INJECTED_ERROR,
    make_intent_task,
    register_variant_domain,
)
from tau2.lifecycle.world import (
    branch_id,
    choose_task,
    instantiate_task,
    make_customers,
    open_ledger,
    project_state,
    register_customers,
    write,
)


class Variant(BaseModel):
    name: str
    agent_llm: str
    failure_rate: float = 0.0


class CampaignConfig(BaseModel):
    """Everything that shapes the data. Its hash stamps the ledger (item 7)."""

    name: str
    seed: int = 7
    customers: int = 30
    days: int = 60
    variants: list[Variant]
    user_llm: str = "gpt-4.1-mini"
    csat_llm: str = "gpt-4.1-mini"
    budget_usd: float = 75.0
    cost_bound_usd: float = 0.25  # reserved per conversation before it runs (OV10)
    max_concurrency: int = 8
    max_steps: int = 100
    task_split: str = "base"
    max_infra_error_share: float = 0.5  # above this, stop the day instead of recording
    hazard: HazardConfig = HazardConfig()

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


Runner = Callable[[Task, Variant, str, str], ConversationOutcome]


# ---------------------------------------------------------------- live runner


def live_runner(cfg: CampaignConfig) -> Runner:
    """Run one real τ²-bench conversation (LLM agent + LLM user) and grade it."""
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.runner.build import build_agent, build_user
    from tau2.runner.simulation import run_simulation

    def run(
        task: Task, variant: Variant, conv_id: str, intent_json: str
    ) -> ConversationOutcome:
        domain = register_variant_domain(variant.failure_rate, f"{cfg.seed}")
        env = registry.get_env_constructor(domain)()
        intent = json.loads(intent_json)
        try:
            agent = build_agent("llm_agent", env, llm=variant.agent_llm, task=task)
            user = build_user("user_simulator", env, task, llm=cfg.user_llm)
            orch = Orchestrator(
                domain=domain,
                agent=agent,
                user=user,
                environment=env,
                task=task,
                max_steps=cfg.max_steps,
                seed=cfg.seed,
                simulation_id=conv_id,
            )
            sim = run_simulation(orch)
        except (
            Exception
        ) as e:  # provider/infra failure: recorded, never retried blindly
            return outcome_from_env(
                env,
                conversation_id=conv_id,
                task_id=task.id,
                infra_error=True,
                transcript=f"infra error: {e}",
                **intent,
            )
        messages = sim.messages or []
        tool_msgs = [
            m
            for m in messages
            if getattr(m, "role", "") == "tool" and m.requestor == "assistant"
        ]
        infra = sim.termination_reason.value in (
            "infrastructure_error",
            "unexpected_error",
        )
        return outcome_from_env(
            env,
            conversation_id=conv_id,
            task_id=task.id,
            task_success=None if infra else sim.reward_info.reward,
            tool_errors=sum(1 for m in tool_msgs if m.error),
            turns=len(messages),
            infra_error=infra,
            cost_usd=(sim.agent_cost or 0.0) + (sim.user_cost or 0.0),
            transcript=_render(messages),
            **intent,
        )

    return run


def _render(messages) -> str:
    lines = []
    for m in messages:
        role = getattr(m, "role", "?")
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                lines.append(f"{role} -> {tc.name}({json.dumps(tc.arguments)})")
        if getattr(m, "content", None):
            lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------- campaign loop


def run_campaign(
    cfg: CampaignConfig,
    out_dir: Path,
    runner: Optional[Runner] = None,
    csat_fn: Optional[CsatFn] = None,
    templates: Optional[list[Task]] = None,
) -> dict:
    """Run (or resume) a campaign in `out_dir`. Returns a status dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = open_ledger(out_dir / "ledger.sqlite")
    _claim_config(conn, cfg, out_dir)
    runner = runner or live_runner(cfg)
    csat_fn = csat_fn or llm_csat_rater(cfg.csat_llm)
    if templates is None:
        from tau2.domains.telecom.environment import get_tasks

        templates = get_tasks(cfg.task_split)
    customers = make_customers(cfg.customers, cfg.seed)
    traits = {c.pop_id: c.traits for c in customers}
    variants = {v.name: v for v in cfg.variants}
    register_customers(conn, customers, list(variants))
    branches = [branch_id(c.pop_id, v) for c in customers for v in variants]

    day = int(_meta(conn, "last_completed_day", -1)) + 1
    if not _retry_pending_csat(conn, csat_fn, traits):
        return _status(conn, "csat_pending", day)
    while day < cfg.days:
        if (out_dir / "CANCEL").exists():
            return _status(conn, "cancelled", day)
        due = _due(conn, branches, day)
        spent = _spent(conn)
        if spent + cfg.cost_bound_usd * len(due) > cfg.budget_usd:
            return _status(conn, "budget_stopped", day, spent=spent)
        jobs = []
        for b, contact in due:
            state = project_state(conn, b, day)
            pop_id, vname = b.split(":", 1)
            rng = _rng(cfg.seed, pop_id, day, "task")
            if contact["intent"]:
                task = instantiate_task(
                    make_intent_task(
                        state, contact["intent"], bool(contact["accepts_offer"])
                    ),
                    state,
                )
                swaps = []
            else:
                task, swaps = choose_task(templates, state, rng)
            conv_id = f"{b}:d{day}"
            task = task.model_copy(update={"id": f"{conv_id}|{task.id}"})
            intent = json.dumps(
                {"intent": contact["intent"], "accepts_offer": contact["accepts_offer"]}
            )
            jobs.append((b, conv_id, task, variants[vname], intent, swaps))
        with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as pool:
            outcomes = list(pool.map(lambda j: runner(j[2], j[3], j[1], j[4]), jobs))
        infra = [o for o in outcomes if o.infra_error]
        if outcomes and len(infra) > cfg.max_infra_error_share * len(outcomes):
            # Provider outage: record nothing for today so resume re-runs the day.
            return _status(conn, "infra_stopped", day, error=infra[0].transcript[:300])
        pending = False
        for (b, conv_id, task, variant, _, swaps), outcome in zip(jobs, outcomes):
            if swaps:
                write(
                    conn,
                    events=[
                        {
                            "event_id": f"{conv_id}:swaps",
                            "branch_id": b,
                            "virtual_day": day,
                            "type": "template_swapped",
                            "payload": {"swaps": swaps},
                        }
                    ],
                )
            res = record_conversation(
                conn, b, day, outcome, traits[b.split(":")[0]], csat_fn, cfg.hazard
            )
            pending |= res["pending"]
            _record_injections(conn, b, day, conv_id, outcome)
        if pending:
            return _status(conn, "csat_pending", day)
        for b in branches:
            daily_tick(conn, b, day, traits[b.split(":")[0]], cfg.seed, cfg.hazard)
        _set_meta(conn, "last_completed_day", day)
        day += 1
    return _status(conn, "complete", day)


def _due(
    conn: sqlite3.Connection, branches: list[str], day: int
) -> list[tuple[str, dict]]:
    """At most one conversation per branch per day (OV5); extra contacts move to tomorrow."""
    due = []
    for b in branches:
        state = project_state(conn, b, day)
        if state["churned"]:
            continue
        rows = conn.execute(
            "SELECT event_id, payload FROM events WHERE branch_id=? AND type='contact_scheduled' ORDER BY rowid",
            (b,),
        ).fetchall()
        today = [
            (eid, json.loads(p)) for eid, p in rows if json.loads(p)["due_day"] == day
        ]
        done = conn.execute(
            "SELECT 1 FROM events WHERE event_id=?", (f"{b}:{b}:d{day}:conversation",)
        ).fetchone()
        if not today or done:
            continue
        due.append((b, today[0][1]))
        for eid, p in today[1:]:
            write(
                conn,
                events=[
                    {
                        "event_id": f"{eid}:deferred",
                        "branch_id": b,
                        "virtual_day": day,
                        "type": "contact_scheduled",
                        "payload": {**p, "due_day": day + 1, "deferred": True},
                    }
                ],
            )
    return due


def _retry_pending_csat(conn, csat_fn: CsatFn, traits: dict) -> bool:
    """Rate calls whose CSAT is still pending from an earlier run. True when none remain."""
    rows = conn.execute(
        "SELECT branch_id, virtual_day, payload FROM events WHERE type='csat_pending'"
    ).fetchall()
    ok = True
    for b, day, p in rows:
        cid = json.loads(p)["conversation_id"]
        if conn.execute(
            "SELECT 1 FROM events WHERE event_id=?", (f"{b}:{cid}:csat",)
        ).fetchone():
            continue
        (payload,) = conn.execute(
            "SELECT payload FROM events WHERE event_id=?", (f"{b}:{cid}:conversation",)
        ).fetchone()
        outcome = ConversationOutcome.model_validate_json(payload)
        try:
            score, comment = csat_fn(outcome, traits[b.split(":")[0]])
        except Exception:
            ok = False
            continue
        write(
            conn,
            events=[
                {
                    "event_id": f"{b}:{cid}:csat",
                    "branch_id": b,
                    "virtual_day": day,
                    "type": "csat",
                    "payload": {
                        "conversation_id": cid,
                        "score": score,
                        "comment": comment,
                    },
                }
            ],
        )
    return ok


def _record_injections(
    conn, branch, day, conv_id, outcome: ConversationOutcome
) -> None:
    n = outcome.transcript.count(INJECTED_ERROR)
    if n:
        write(
            conn,
            oracle=[
                {
                    "oracle_id": f"{conv_id}:injections",
                    "branch_id": branch,
                    "virtual_day": day,
                    "kind": "injected_failures",
                    "payload": {"conversation_id": conv_id, "count": n},
                }
            ],
        )


def _claim_config(conn, cfg: CampaignConfig, out_dir: Path) -> None:
    existing = _meta(conn, "config_hash", None)
    if existing is None:
        _set_meta(conn, "config_hash", cfg.digest())
        (out_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    elif existing != cfg.digest():
        raise ValueError(
            f"Ledger in {out_dir} was created with config {existing}; this config is "
            f"{cfg.digest()}. Start a new output directory instead of resuming."
        )


def _spent(conn) -> float:
    return sum(
        json.loads(p).get("cost_usd", 0.0)
        for (p,) in conn.execute("SELECT payload FROM events WHERE type='conversation'")
    )


def _status(conn, status: str, day: int, **extra) -> dict:
    _set_meta(conn, "status", status)
    return {"status": status, "day": day, "spent_usd": round(_spent(conn), 4), **extra}


def _meta(conn, key, default):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def _set_meta(conn, key, value) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value))
        )
