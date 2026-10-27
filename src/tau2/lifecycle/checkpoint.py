"""Lifecycle checkpoints: what a customer does after (and between) conversations.

Two entry points, both writing only through the ledger:

- `record_conversation`: after a call, persist what the agent changed (plan,
  line, credit), classify intent calls (OV1), get a CSAT rating, and log the
  call's direct hazard contribution as private oracle data (D7).
- `daily_tick`: once per virtual day per branch, the authored hazard (D5)
  decides whether the customer wants to leave (cancel or downgrade call),
  returns for help after a failed call, and how much they use the product.

Random draws are keyed by (population customer, day, purpose), not by branch,
so all agent variants see the same draws until their histories diverge.
See .plans/tau-lifecycle-orm.md (D5, D7) and .plans/telecom-lifecycle-tools.md.
"""

import hashlib
import json
import math
import random
import sqlite3
from typing import Callable, Optional

from pydantic import BaseModel

from tau2.lifecycle.world import (
    TEMPLATE_CUSTOMER_ID,
    TEMPLATE_LINE_ID,
    project_state,
    write,
)

PLAN_ORDER = {"P1001": 0, "P1002": 1, "P1003": 2}  # Basic < Premium < Unlimited Plus
CSAT_ATTEMPTS = 3


class HazardConfig(BaseModel):
    """Authored hazard weights. Lives in the versioned campaign config (item 7).

    Defaults give ~7% leave intent per 30 days with no bad calls (400-customer
    probe, 2026-10-27), inside the D15 3-10% band; the pilot re-tunes them.
    """

    w_csat: float = 0.6  # per CSAT point below 3
    w_fail: float = 1.0  # native task failure
    w_tool_err: float = 0.3  # scaled by min(tool errors, 3) / 3
    w_repeat: float = 0.4  # per contact beyond the first in the last 30 days
    w_price: float = 0.3  # price sensitivity while on a plan above Basic
    decay_days: float = 14.0
    noise_sd: float = 0.2
    downgrade_share: float = 0.5  # share of leave intents that downgrade instead
    p_return_after_fail: float = 0.6  # scaled by (1 - patience)
    suppression_days: int = 30  # OV2: accepted offer suppresses cancel intent
    usage_every_days: int = 7


class ConversationOutcome(BaseModel):
    """What one call produced. Built from a τ²-bench simulation (or a scripted run)."""

    conversation_id: str
    task_id: str
    intent: Optional[str] = None  # "downgrade" | "upgrade" | "cancel" | None (support)
    accepts_offer: Optional[bool] = None
    task_success: Optional[float] = None  # native reward; None on infra error
    tool_errors: int = 0
    turns: int = 0
    infra_error: bool = False
    cost_usd: float = 0.0
    transcript: str = ""
    # Agent DB after the call (template IDs C1001/L1002 inside the branch's world)
    plan_id: str
    line_status: str
    goodwill_used: float


CsatFn = Callable[[ConversationOutcome, dict], tuple[int, str]]


def outcome_from_env(env, **fields) -> ConversationOutcome:
    """Read the post-call agent DB of a telecom(_lifecycle) environment."""
    db = env.tools.db
    line = next(ln for ln in db.lines if ln.line_id == TEMPLATE_LINE_ID)
    customer = next(c for c in db.customers if c.customer_id == TEMPLATE_CUSTOMER_ID)
    return ConversationOutcome(
        plan_id=line.plan_id,
        line_status=line.status.value,
        goodwill_used=customer.goodwill_credit_used_this_year,
        **fields,
    )


# ---------------------------------------------------------------- after a call


def record_conversation(
    conn: sqlite3.Connection,
    branch: str,
    day: int,
    outcome: ConversationOutcome,
    traits: dict,
    csat_fn: CsatFn,
    cfg: HazardConfig = HazardConfig(),
) -> dict:
    """Persist one call and its consequences in a single transaction.

    Returns {"intent_outcome", "csat", "pending"}. A pending CSAT means the
    rater failed CSAT_ATTEMPTS times; the caller must not advance this branch
    past `day` until it is resolved (OV7). No default rating is ever written.
    """
    before = project_state(conn, branch, day)
    cid = outcome.conversation_id
    events = [
        _ev(
            branch,
            day,
            f"{cid}:conversation",
            "conversation",
            outcome.model_dump(exclude={"transcript"}),
        ),
    ]

    # Persist what the agent changed (OV7)
    if outcome.plan_id != before["plan_id"]:
        direction = (
            "down"
            if PLAN_ORDER.get(outcome.plan_id, 1) < PLAN_ORDER.get(before["plan_id"], 1)
            else "up"
        )
        events.append(
            _ev(
                branch,
                day,
                f"{cid}:plan",
                "plan_changed",
                {"plan_id": outcome.plan_id, "direction": direction, "by": "agent"},
            )
        )
    credit = round(outcome.goodwill_used - before["goodwill_credit_used_this_year"], 2)
    if credit > 0:
        events.append(
            _ev(
                branch,
                day,
                f"{cid}:credit",
                "goodwill_credit_applied",
                {"amount": credit},
            )
        )
    closed = outcome.line_status == "Closed"
    if closed:
        events += _churn(branch, day, cid, by="agent")

    # Intent calls get an explicit outcome from the customer's decision (OV1)
    intent_outcome = None
    if outcome.intent == "cancel":
        if closed:
            intent_outcome = "cancelled"
        elif outcome.accepts_offer and credit > 0:
            intent_outcome = "retained"
        else:
            intent_outcome = "cancel_blocked"
            events += _churn(branch, day + 1, cid, by="self_serve")
    elif outcome.intent in ("downgrade", "upgrade"):
        wanted = "P1001" if outcome.intent == "downgrade" else "P1003"
        if outcome.plan_id == wanted:
            intent_outcome = "changed"
        else:
            intent_outcome = "change_blocked"
            direction = "down" if outcome.intent == "downgrade" else "up"
            events.append(
                _ev(
                    branch,
                    day + 1,
                    f"{cid}:self_plan",
                    "plan_changed",
                    {"plan_id": wanted, "direction": direction, "by": "self_serve"},
                )
            )
    if intent_outcome:
        events.append(
            _ev(
                branch,
                day,
                f"{cid}:intent",
                "intent_outcome",
                {
                    "conversation_id": cid,
                    "intent": outcome.intent,
                    "outcome": intent_outcome,
                },
            )
        )

    # CSAT: forced to 1 when the agent blocked what the customer asked for or the
    # call broke; otherwise the customer-perspective LLM rater decides.
    csat, comment, pending = None, "", False
    if outcome.infra_error or intent_outcome in ("cancel_blocked", "change_blocked"):
        csat, comment = 1, "forced: request not handled"
    else:
        for _ in range(CSAT_ATTEMPTS):
            try:
                csat, comment = csat_fn(outcome, traits)
                break
            except Exception as e:  # rater/provider failure
                comment = f"rater failed: {e}"
        pending = csat is None
    if pending:
        events.append(
            _ev(
                branch,
                day,
                f"{cid}:csat_pending",
                "csat_pending",
                {"conversation_id": cid, "reason": comment},
            )
        )
    else:
        events.append(
            _ev(
                branch,
                day,
                f"{cid}:csat",
                "csat",
                {"conversation_id": cid, "score": csat, "comment": comment},
            )
        )

    oracle = [
        {
            "oracle_id": f"{branch}:{cid}:direct_contribution",
            "branch_id": branch,
            "virtual_day": day,
            "kind": "direct_contribution",
            "payload": {
                "conversation_id": cid,
                "logit": _call_term(
                    csat, outcome.task_success, outcome.tool_errors, cfg
                ),
            },
        }
    ]
    # One transaction; deterministic ids make a replay after a crash a no-op.
    write(conn, events=events, oracle=oracle)
    return {"intent_outcome": intent_outcome, "csat": csat, "pending": pending}


def _churn(branch, day, cid, by):
    return [
        _ev(branch, day, f"{cid}:line_cancelled:{by}", "line_cancelled", {"by": by}),
        _ev(branch, day, f"{cid}:churned:{by}", "churned", {"by": by}),
    ]


def _call_term(csat, task_success, tool_errors, cfg: HazardConfig) -> float:
    """Undecayed logit a call adds to the leave hazard (the D7 oracle quantity)."""
    term = cfg.w_tool_err * min(tool_errors, 3) / 3
    if csat is not None:
        term += cfg.w_csat * max(0, 3 - csat)
    if task_success is not None:
        term += cfg.w_fail * (1 - task_success)
    return round(term, 4)


# ---------------------------------------------------------------- every day


def daily_tick(
    conn: sqlite3.Connection,
    branch: str,
    day: int,
    traits: dict,
    seed: int,
    cfg: HazardConfig = HazardConfig(),
) -> list[dict]:
    """Decide today's silent behaviour and any calls to schedule. Returns events written."""
    state = project_state(conn, branch, day)
    if state["churned"]:
        return []
    pop_id = branch.split(":")[0]
    rows = _events(conn, branch, day)
    convs = [(d, p) for d, t, p in rows if t == "conversation"]
    csats = {p["conversation_id"]: p["score"] for _, t, p in rows if t == "csat"}
    scheduled = [d for d, t, p in rows if t == "contact_scheduled"]
    pending_contact = bool(scheduled) and max(scheduled) >= (
        convs[-1][0] if convs else -1
    )
    retained = [
        d for d, t, p in rows if t == "intent_outcome" and p["outcome"] == "retained"
    ]

    # Hazard (D5): base churn + decayed call terms + repeat contacts + price + noise
    daily_base = 1 - (1 - traits["base_churn"]) ** (1 / 30)
    call_terms = sum(
        _call_term(
            csats.get(p["conversation_id"]), p["task_success"], p["tool_errors"], cfg
        )
        * math.exp(-(day - d) / cfg.decay_days)
        for d, p in convs
        if day - d <= 60
    )
    recent = sum(1 for d, _ in convs if day - d <= 30)
    repeat_term = cfg.w_repeat * max(0, recent - 1)
    price_term = (
        cfg.w_price * traits["price_sensitivity"] * (state["plan_id"] != "P1001")
    )
    noise = _rng(seed, pop_id, day, "noise").gauss(0, cfg.noise_sd)
    logit = (
        math.log(daily_base / (1 - daily_base))
        + call_terms
        + repeat_term
        + price_term
        + noise
    )
    p_leave = 1 / (1 + math.exp(-logit))
    suppressed = any(day - d < cfg.suppression_days for d in retained)
    if suppressed:
        p_leave = 0.0

    events, decision = [], None
    if not pending_contact:
        if _rng(seed, pop_id, day, "leave").random() < p_leave:
            downgrade = (
                state["plan_id"] != "P1001"
                and _rng(seed, pop_id, day, "how").random()
                < cfg.downgrade_share * 2 * traits["price_sensitivity"]
            )
            decision = "downgrade" if downgrade else "cancel"
        elif (
            convs
            and convs[-1][0] == day - 1
            and (convs[-1][1]["task_success"] or 0) < 1
        ):
            if _rng(seed, pop_id, day, "return").random() < cfg.p_return_after_fail * (
                1 - traits["patience"]
            ):
                decision = "support"
    if decision:
        events.append(
            _ev(
                branch,
                day,
                f"{day}:contact",
                "contact_scheduled",
                {
                    "due_day": day + 1,
                    "intent": None if decision == "support" else decision,
                    "accepts_offer": traits["price_sensitivity"] > 0.5
                    if decision == "cancel"
                    else None,
                },
            )
        )
    if day % cfg.usage_every_days == 0:
        sessions = max(
            0,
            round(
                _rng(seed, pop_id, day, "usage").gauss(10 * math.exp(-call_terms), 2)
            ),
        )
        events.append(_ev(branch, day, f"{day}:usage", "usage", {"sessions": sessions}))

    oracle = [
        {
            "oracle_id": f"{branch}:{day}:hazard",
            "branch_id": branch,
            "virtual_day": day,
            "kind": "hazard",
            "payload": {
                "logit": round(logit, 4),
                "p_leave": p_leave,
                "call_terms": round(call_terms, 4),
                "repeat_term": repeat_term,
                "price_term": price_term,
                "noise": round(noise, 4),
                "suppressed": suppressed,
                "decision": decision,
            },
        }
    ]
    write(conn, events=events, oracle=oracle)
    return events


# ---------------------------------------------------------------- helpers


def _ev(branch, day, eid, etype, payload) -> dict:
    return {
        "event_id": f"{branch}:{eid}",
        "branch_id": branch,
        "virtual_day": day,
        "type": etype,
        "payload": payload,
    }


def _events(conn, branch, day):
    return [
        (d, t, json.loads(p))
        for d, t, p in conn.execute(
            "SELECT virtual_day, type, payload FROM events WHERE branch_id = ? AND virtual_day <= ? ORDER BY virtual_day, rowid",
            (branch, day),
        )
    ]


def _rng(seed: int, pop_id: str, day: int, purpose: str) -> random.Random:
    key = hashlib.sha256(f"{seed}:{pop_id}:{day}:{purpose}".encode()).digest()
    return random.Random(int.from_bytes(key[:8], "big"))


def llm_csat_rater(model: str) -> CsatFn:
    """Customer-perspective CSAT via LiteLLM. Raises on bad output so the caller retries."""
    import litellm

    def rate(outcome: ConversationOutcome, traits: dict) -> tuple[int, str]:
        prompt = (
            "You are the customer in this support conversation. Rate your experience "
            "from 1 (very dissatisfied) to 5 (very satisfied) as a real customer would, "
            f"given your patience ({traits['patience']:.2f} on 0-1). "
            'Answer only JSON: {"score": <1-5>, "comment": "<one sentence>"}\n\n'
            + outcome.transcript
        )
        resp = litellm.completion(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        data = json.loads(resp.choices[0].message.content)
        score = int(data["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"CSAT out of range: {score}")
        return score, str(data.get("comment", ""))

    return rate
