"""Lifecycle ledger: the single source of truth for customers across conversations.

`events` holds everything a real company could observe (conversations, CSAT,
lifecycle actions). `oracle` holds private simulator truth (hidden traits,
hazard terms, injected failures) and is never read by exports or the default UI.
A branch is one population customer under one agent variant.
"""

import json
import random
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel

from tau2.data_model.tasks import EnvFunctionCall, InitializationData, Task
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.environment import get_environment
from tau2.domains.telecom.utils import TELECOM_DB_PATH

SCHEMA_VERSION = 1

# Every upstream telecom task is written for this customer and line.
TEMPLATE_CUSTOMER_ID = "C1001"
TEMPLATE_LINE_ID = "L1002"
TEMPLATE_NAME = "John Smith"
DEFAULT_PLAN_ID = "P1002"
PERSONA_LEVELS = ("None", "Easy", "Hard")
BILLING_CYCLE_DAYS = 30
CREDIT_YEAR_DAYS = 365
_NETWORK_SEARCH = EnvFunctionCall(
    env_type="user", func_name="simulate_network_search", arguments={}
)

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


# ---------------------------------------------------------------- ledger I/O


def branch_id(pop_id: str, variant: str) -> str:
    """One population customer under one agent variant."""
    return f"{pop_id}:{variant}"


def append_events(conn: sqlite3.Connection, events: Iterable[dict]) -> None:
    """Write events in one transaction; an event_id already present is skipped."""
    write(conn, events=events)


def append_oracle(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    """Write private simulator rows in one transaction; replays are no-ops."""
    write(conn, oracle=rows)


def write(
    conn: sqlite3.Connection, events: Iterable[dict] = (), oracle: Iterable[dict] = ()
) -> None:
    """Write public events and private oracle rows atomically (one transaction)."""
    with conn:
        _insert(conn, "events", "event_id", "type", events)
        _insert(conn, "oracle", "oracle_id", "kind", oracle)


def _insert(conn, table, id_col, kind_col, rows) -> None:
    conn.executemany(
        f"INSERT OR IGNORE INTO {table} ({id_col}, branch_id, virtual_day, {kind_col}, payload)"
        " VALUES (?, ?, ?, ?, ?)",
        [
            (
                r[id_col],
                r["branch_id"],
                r["virtual_day"],
                r[kind_col],
                json.dumps(r.get("payload", {})),
            )
            for r in rows
        ],
    )


# ---------------------------------------------------------------- customers


class Customer(BaseModel):
    """A population customer. `traits` are private and go to the oracle table."""

    pop_id: str
    full_name: str
    date_of_birth: str
    email: str
    persona: str
    traits: dict[str, float]


_FIRST = [
    "Dana",
    "Luis",
    "Priya",
    "Marcus",
    "Aiko",
    "Fatima",
    "Owen",
    "Grace",
    "Tomas",
    "Nia",
    "Ravi",
    "Elena",
    "Kwame",
    "Sofia",
    "Jonah",
    "Mei",
]
_LAST = [
    "Ortiz",
    "Nguyen",
    "Patel",
    "Brooks",
    "Tanaka",
    "Haddad",
    "Reilly",
    "Kim",
    "Silva",
    "Okafor",
    "Iyer",
    "Rossi",
    "Mensah",
    "Garcia",
    "Weiss",
    "Chen",
]


def make_customers(n: int, seed: int) -> list[Customer]:
    """Deterministic synthetic population. Persona uses telecom's 3 upstream levels.

    ponytail: name lists + uniform traits; swap in MatrAIx persona sampling if the
    pilot shows customers are too uniform.
    """
    rng = random.Random(seed)
    customers = []
    for i in range(n):
        first, last = rng.choice(_FIRST), rng.choice(_LAST)
        year, month, day = (
            rng.randint(1950, 2002),
            rng.randint(1, 12),
            rng.randint(1, 28),
        )
        customers.append(
            Customer(
                pop_id=f"P{i + 1:04d}",
                full_name=f"{first} {last}",
                date_of_birth=f"{year}-{month:02d}-{day:02d}",
                email=f"{first.lower()}.{last.lower()}{i + 1}@example.com",
                persona=rng.choice(PERSONA_LEVELS),
                traits={
                    "patience": round(rng.random(), 3),
                    "price_sensitivity": round(rng.random(), 3),
                    "base_churn": round(rng.betavariate(2, 30), 4),
                },
            )
        )
    return customers


def register_customers(
    conn: sqlite3.Connection, customers: list[Customer], variants: list[str]
) -> None:
    """Day-0 profile event per branch (public) and traits row (private)."""
    events, oracle = [], []
    for c in customers:
        profile = c.model_dump(exclude={"traits"})
        for v in variants:
            b = branch_id(c.pop_id, v)
            events.append(
                {
                    "event_id": f"{b}:0:customer_created",
                    "branch_id": b,
                    "virtual_day": 0,
                    "type": "customer_created",
                    "payload": {**profile, "plan_id": DEFAULT_PLAN_ID},
                }
            )
            oracle.append(
                {
                    "oracle_id": f"{b}:0:traits",
                    "branch_id": b,
                    "virtual_day": 0,
                    "kind": "traits",
                    "payload": c.traits,
                }
            )
    write(conn, events=events, oracle=oracle)


# ---------------------------------------------------------------- state projection


def project_state(conn: sqlite3.Connection, branch: str, day: int) -> dict:
    """Customer state as of `day`, rebuilt from public events only."""
    rows = conn.execute(
        "SELECT virtual_day, type, payload FROM events"
        " WHERE branch_id = ? AND virtual_day <= ? ORDER BY virtual_day, rowid",
        (branch, day),
    ).fetchall()
    if not rows:
        raise KeyError(f"No events for branch {branch}")
    state: dict = {"account_status": "Active", "churned": False, "history": []}
    for vday, etype, payload in rows:
        p = json.loads(payload)
        if etype == "customer_created":
            state.update(p)
        elif etype == "plan_changed":
            state["plan_id"] = p["plan_id"]
        elif etype == "account_status_changed":
            state["account_status"] = p["status"]
        elif etype == "churned":
            state["churned"] = True
        elif etype == "goodwill_credit_applied" and vday // CREDIT_YEAR_DAYS == (
            day // CREDIT_YEAR_DAYS
        ):
            state["goodwill_credits_this_year"] = (
                state.get("goodwill_credits_this_year", 0.0) + p["amount"]
            )
        state["history"].append({"day": vday, "type": etype})
    # Billing cycle (D6/OV5): usage resets every 30 days, never on a plan change.
    # ponytail: no usage growth model yet, so after the first reset usage stays 0.
    if day >= BILLING_CYCLE_DAYS:
        state["data_used_gb"] = 0.0
        state["data_refueling_gb"] = 0.0
    # Template customer starts the year with 25.0 used; the counter resets yearly.
    base_credit = _template_customer()["goodwill_credit_used_this_year"]
    state["goodwill_credit_used_this_year"] = (
        base_credit if day < CREDIT_YEAR_DAYS else 0.0
    ) + state.pop("goodwill_credits_this_year", 0.0)
    return state


# ---------------------------------------------------------------- tasks


@lru_cache(maxsize=1)
def _base_db() -> dict:
    return TelecomDB.load(TELECOM_DB_PATH).model_dump(mode="json")


def _template_customer() -> dict:
    return next(
        c for c in _base_db()["customers"] if c["customer_id"] == TEMPLATE_CUSTOMER_ID
    )


def instantiate_task(template: Task, state: dict) -> Task:
    """Rewrite a C1001 telecom task for one customer branch.

    Customer IDs stay C1001/L1002 inside each branch's own world; only personal
    fields and account state change. History goes in initialization_data, which
    set_state applies before the template's initialization_actions, so the
    task's own issue setup wins where they overlap. Predicted and gold
    environments both replay the same init, so native grading stays valid.
    """
    if template.initial_state and template.initial_state.initialization_data:
        raise NotImplementedError(f"{template.id} already sets initialization_data")
    data = json.loads(
        template.model_dump_json().replace(TEMPLATE_NAME, state["full_name"])
    )
    db = _base_db()
    customers = [
        {
            **c,
            "full_name": state["full_name"],
            "date_of_birth": state["date_of_birth"],
            "email": state["email"],
            "account_status": state["account_status"],
            "goodwill_credit_used_this_year": state.get(
                "goodwill_credit_used_this_year", c["goodwill_credit_used_this_year"]
            ),
        }
        if c["customer_id"] == TEMPLATE_CUSTOMER_ID
        else c
        for c in db["customers"]
    ]
    usage = {k: state[k] for k in ("data_used_gb", "data_refueling_gb") if k in state}
    lines = [
        {**ln, "plan_id": state["plan_id"], **usage}
        if ln["line_id"] == TEMPLATE_LINE_ID
        else ln
        for ln in db["lines"]
    ]
    init = data.get("initial_state") or {}
    init["initialization_data"] = InitializationData(
        agent_data={"customers": customers, "lines": lines}
    ).model_dump(mode="json")
    data["initial_state"] = init
    return Task.model_validate(data)


def check_applicable(task: Task, env_factory=get_environment) -> Optional[str]:
    """None if the gold path still solves `task`; otherwise the reason it doesn't.

    Replays gold actions on a fresh environment, lets the user's phone re-scan
    the network (as it does when the simulated user next checks it), then checks
    every env assertion. A few upstream templates fail this check unmodified;
    `choose_task` never uses those.
    """
    env = env_factory()
    init = task.initial_state
    env.set_state(
        initialization_data=init.initialization_data if init else None,
        initialization_actions=init.initialization_actions if init else None,
        message_history=[],
    )
    for action in task.evaluation_criteria.actions or []:
        try:
            env.make_tool_call(
                tool_name=action.name, requestor=action.requestor, **action.arguments
            )
        except Exception as e:
            return f"gold action {action.name} failed: {e}"
    env.run_env_function_call(_NETWORK_SEARCH)
    for assertion in task.evaluation_criteria.env_assertions or []:
        if not env.run_env_assertion(assertion, raise_assertion_error=False):
            return f"assertion {assertion.func_name} not met on gold path"
    return None


def choose_task(
    templates: list[Task], state: dict, rng: random.Random
) -> tuple[Task, list[dict]]:
    """Pick an applicable template for this customer; return it and any swaps made.

    Templates are filtered to the customer's persona level, then tried in a
    seeded random order. Each rejected template is reported so the caller can
    log a template_swapped event.
    """
    suffix = f"[PERSONA:{state['persona']}]"
    pool = [t for t in templates if t.id.endswith(suffix) and _template_ok(t)]
    rng.shuffle(pool)
    swaps = []
    for template in pool:
        task = instantiate_task(template, state)
        reason = check_applicable(task)
        if reason is None:
            return task, swaps
        swaps.append({"template_id": template.id, "reason": reason})
    raise LookupError(f"No applicable template for persona {state['persona']}")


_TEMPLATE_OK: dict[str, bool] = {}


def _template_ok(template: Task) -> bool:
    """Whether the unmodified upstream template passes the gold-path check."""
    if template.id not in _TEMPLATE_OK:
        _TEMPLATE_OK[template.id] = check_applicable(template) is None
    return _TEMPLATE_OK[template.id]
