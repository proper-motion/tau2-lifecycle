"""Exports from the ledger: ORM training rows, private evaluation data, analytics.

Training rows are built field by field from an allowlist (R5): opaque ids, the
virtual day, the agent-visible transcript and mature window labels. Nothing
else from the ledger reaches them: no variant (it encodes the agent model and
injected failure rate), no task id (it names the hidden scenario), no intent or
offer decision, no native reward, no oracle data. Private evaluation data
(direct hazard contribution, variant, native reward, injected-failure count) is
written to a separate file keyed by the same row id, for scoring only (D7).

Analytics events go to an isolated synthetic Mixpanel or Amplitude project,
with stable insert ids and virtual timestamps. The ledger stays authoritative:
sending never writes to it.
"""

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import requests

from tau2.lifecycle.kpis import window_labels

# Virtual day 0 = the telecom scenario clock (main_policy.md: 2025-02-25 12:08 EST)
VIRTUAL_EPOCH = datetime(2025, 2, 25, 17, 8, tzinfo=timezone.utc)

TRAIN_FIELDS = (
    "row_id",
    "customer_key",
    "virtual_day",
    "transcript",
    "churn_or_downgrade_28d",
    "repeat_contact_7d",
    "csat",
)


def _key(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:16]


def _salt(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key='config_hash'").fetchone()
    return row[0] if row else "unsalted"


def _conversations(conn: sqlite3.Connection, cutoff_day: int) -> dict[str, dict]:
    return {
        (p := json.loads(payload))["conversation_id"]: {**p, "branch_id": b, "day": d}
        for b, d, payload in conn.execute(
            "SELECT branch_id, virtual_day, payload FROM events"
            " WHERE type='conversation' AND virtual_day <= ?",
            (cutoff_day,),
        )
    }


def training_rows(conn: sqlite3.Connection, cutoff_day: int) -> list[dict]:
    """Mature, non-infra conversations with permitted fields only."""
    salt = _salt(conn)
    convs = _conversations(conn, cutoff_day)
    rows = []
    for label in window_labels(conn, cutoff_day):
        conv = convs[label["conversation_id"]]
        if conv["infra_error"] or label["churn_or_downgrade_28d"] is None:
            continue  # infra errors and open windows never become training rows
        pop_id = conv["branch_id"].split(":", 1)[0]
        row = {
            "row_id": _key(salt, label["conversation_id"]),
            "customer_key": _key(
                salt, pop_id
            ),  # same for all variants: split by customer
            "virtual_day": conv["day"],
            "transcript": conv["transcript"],
            "churn_or_downgrade_28d": label["churn_or_downgrade_28d"],
            "repeat_contact_7d": label["repeat_contact_7d"],
            "csat": label["csat"],
        }
        assert tuple(row) == TRAIN_FIELDS
        rows.append(row)
    return rows


def private_eval_rows(conn: sqlite3.Connection, cutoff_day: int) -> list[dict]:
    """Scoring-only data keyed by row_id. Never an input to a judge or the ORM."""
    salt = _salt(conn)
    oracle = {}
    for kind, payload in conn.execute(
        "SELECT kind, payload FROM oracle WHERE kind IN ('direct_contribution', 'injected_failures')"
    ):
        p = json.loads(payload)
        oracle.setdefault(p["conversation_id"], {})[kind] = p
    out = []
    for cid, conv in _conversations(conn, cutoff_day).items():
        o = oracle.get(cid, {})
        out.append(
            {
                "row_id": _key(salt, cid),
                "variant": conv["branch_id"].split(":", 1)[1],
                "task_id": conv["task_id"],
                "intent": conv.get("intent"),
                "task_success": conv["task_success"],
                "infra_error": conv["infra_error"],
                "direct_hazard_contribution": o.get("direct_contribution", {}).get(
                    "logit"
                ),
                "injected_failures": o.get("injected_failures", {}).get("count", 0),
            }
        )
    return out


def write_exports(conn: sqlite3.Connection, out_dir: Path, cutoff_day: int) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "private").mkdir(parents=True, exist_ok=True)
    train, private = (
        training_rows(conn, cutoff_day),
        private_eval_rows(conn, cutoff_day),
    )
    _jsonl(out_dir / "train_rows.jsonl", train)
    _jsonl(out_dir / "private" / "eval_private.jsonl", private)
    return {"train_rows": len(train), "private_rows": len(private)}


def _jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


# ---------------------------------------------------------------- analytics

# Public event type -> (analytics event name, allowed payload properties)
ANALYTICS_EVENTS = {
    "customer_created": ("Customer Created", ("persona",)),
    "conversation": ("Support Conversation", ("turns", "infra_error")),
    "csat": ("CSAT Submitted", ("score",)),
    "intent_outcome": ("Retention Outcome", ("intent", "outcome")),
    "plan_changed": ("Plan Changed", ("plan_id", "direction", "by")),
    "goodwill_credit_applied": ("Credit Applied", ("amount",)),
    "churned": ("Churned", ("by",)),
    "usage": ("Product Usage", ("sessions",)),
}


def analytics_events(conn: sqlite3.Connection, cutoff_day: int) -> list[dict]:
    """Provider-neutral events: allowlisted properties, virtual timestamps, stable ids."""
    salt = _salt(conn)
    events = []
    for event_id, branch, day, etype, payload in conn.execute(
        "SELECT event_id, branch_id, virtual_day, type, payload FROM events"
        " WHERE virtual_day <= ? ORDER BY virtual_day, rowid",
        (cutoff_day,),
    ):
        if etype not in ANALYTICS_EVENTS:
            continue
        name, allowed = ANALYTICS_EVENTS[etype]
        p = json.loads(payload)
        pop_id, variant = branch.split(":", 1)
        props = {k: p[k] for k in allowed if k in p}
        if etype == "conversation":
            props["conversation_key"] = _key(salt, p["conversation_id"])
        events.append(
            {
                "event": name,
                "insert_id": _key(salt, event_id),
                "user_id": _key(salt, branch),
                "time": VIRTUAL_EPOCH + timedelta(days=day),
                "properties": {
                    **props,
                    "customer_key": _key(salt, pop_id),
                    "agent_variant": variant,
                    "virtual_day": day,
                    "synthetic": True,
                },
            }
        )
    return events


def to_mixpanel(events: list[dict]) -> list[dict]:
    return [
        {
            "event": e["event"],
            "properties": {
                **e["properties"],
                "time": int(e["time"].timestamp()),
                "distinct_id": e["user_id"],
                "$insert_id": e["insert_id"],
            },
        }
        for e in events
    ]


def to_amplitude(events: list[dict]) -> list[dict]:
    return [
        {
            "event_type": e["event"],
            "user_id": e["user_id"],
            "time": int(e["time"].timestamp() * 1000),
            "insert_id": e["insert_id"],
            "event_properties": e["properties"],
        }
        for e in events
    ]


def send_mixpanel(
    events: list[dict], project_id: str, username: str, secret: str
) -> int:
    """Mixpanel /import (historical timestamps allowed), batches of 2000."""
    return _send(
        to_mixpanel(events),
        2000,
        lambda batch: requests.post(
            "https://api.mixpanel.com/import",
            params={"strict": "1", "project_id": project_id},
            auth=(username, secret),
            json=batch,
            timeout=30,
        ),
    )


def send_amplitude(events: list[dict], api_key: str) -> int:
    """Amplitude Batch API, batches of 1000."""
    return _send(
        to_amplitude(events),
        1000,
        lambda batch: requests.post(
            "https://api2.amplitude.com/batch",
            json={"api_key": api_key, "events": batch},
            timeout=30,
        ),
    )


def _send(rows: list[dict], batch_size: int, post, retries: int = 4) -> int:
    """POST in batches; retry 429/5xx with backoff; raise on any other error."""
    sent = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        for attempt in range(retries + 1):
            resp: Optional[requests.Response] = post(batch)
            if resp.status_code < 300:
                sent += len(batch)
                break
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < retries:
                    time.sleep(min(2**attempt, 30))
                    continue
            raise RuntimeError(
                f"Analytics send failed at batch {i // batch_size}: "
                f"HTTP {resp.status_code} {resp.text[:300]}"
            )
    return sent
