"""KPIs and outcome labels computed from the public ledger only.

`window_labels` is the single definition of the per-conversation training
labels (D7); exports and the ORM experiment read it, never re-derive it.
A label is None while its window is still open at the cutoff and nothing has
happened yet (immature / right-censored). Native task success, cost and infra
errors are reported separately from business KPIs (item 3).
"""

import json
import sqlite3
from collections import defaultdict
from statistics import mean
from typing import Optional

CHURN_WINDOW_DAYS = 28
REPEAT_WINDOW_DAYS = 7


def _rows(conn: sqlite3.Connection, cutoff_day: int):
    for branch, day, etype, payload in conn.execute(
        "SELECT branch_id, virtual_day, type, payload FROM events"
        " WHERE virtual_day <= ? ORDER BY virtual_day, rowid",
        (cutoff_day,),
    ):
        yield branch, day, etype, json.loads(payload)


def window_labels(conn: sqlite3.Connection, cutoff_day: int) -> list[dict]:
    """One row per conversation: churn_or_downgrade_28d, repeat_contact_7d, csat."""
    convs, leaves, contacts, csat = [], defaultdict(list), defaultdict(list), {}
    for branch, day, etype, p in _rows(conn, cutoff_day):
        if etype == "conversation":
            convs.append((branch, day, p))
            contacts[branch].append(day)
        elif etype == "churned" or (
            etype == "plan_changed" and p.get("direction") == "down"
        ):
            leaves[branch].append(day)
        elif etype == "csat":
            csat[p["conversation_id"]] = p["score"]

    def label(events_days: list[int], start: int, window: int, strict: bool):
        hit = any(
            (start < d if strict else start <= d) and d <= start + window
            for d in events_days
        )
        if hit:
            return 1
        return 0 if start + window <= cutoff_day else None

    out = []
    for branch, day, p in convs:
        cid = p["conversation_id"]
        out.append(
            {
                "conversation_id": cid,
                "branch_id": branch,
                "virtual_day": day,
                "churn_or_downgrade_28d": label(
                    leaves[branch], day, CHURN_WINDOW_DAYS, strict=False
                ),
                "repeat_contact_7d": label(
                    contacts[branch], day, REPEAT_WINDOW_DAYS, strict=True
                ),
                "csat": csat.get(cid),
            }
        )
    return out


def cohort_kpis(conn: sqlite3.Connection, cutoff_day: int) -> dict[str, dict]:
    """Per agent variant. Churn and retention are per customer branch (D17)."""
    by_variant: dict[str, dict] = defaultdict(
        lambda: {
            "branches": set(),
            "churned": set(),
            "csat": [],
            "task_success": [],
            "infra_errors": 0,
            "cost_usd": 0.0,
            "sessions": [],
            "conversations": 0,
        }
    )
    for branch, _, etype, p in _rows(conn, cutoff_day):
        v = by_variant[branch.split(":", 1)[1]]
        v["branches"].add(branch)
        if etype == "churned":
            v["churned"].add(branch)
        elif etype == "csat":
            v["csat"].append(p["score"])
        elif etype == "usage":
            v["sessions"].append(p["sessions"])
        elif etype == "conversation":
            v["conversations"] += 1
            v["cost_usd"] += p.get("cost_usd", 0.0)
            if p.get("infra_error"):
                v["infra_errors"] += 1
            elif p.get("task_success") is not None:
                v["task_success"].append(p["task_success"])

    labels = window_labels(conn, cutoff_day)
    out = {}
    for variant, v in sorted(by_variant.items()):
        mine = [r for r in labels if r["branch_id"].split(":", 1)[1] == variant]
        n = len(v["branches"])
        out[variant] = {
            "customers": n,
            "churn_rate": len(v["churned"]) / n if n else None,
            "retention_rate": 1 - len(v["churned"]) / n if n else None,
            "csat_mean": _mean(v["csat"]),
            "repeat_contact_7d_rate": _mean(
                [
                    r["repeat_contact_7d"]
                    for r in mine
                    if r["repeat_contact_7d"] is not None
                ]
            ),
            "pending_windows": sum(r["churn_or_downgrade_28d"] is None for r in mine),
            "usage_sessions_mean": _mean(v["sessions"]),
            # separate measures, never mixed into business KPIs
            "task_success_rate": _mean(v["task_success"]),
            "infra_errors": v["infra_errors"],
            "cost_usd": round(v["cost_usd"], 4),
            "conversations": v["conversations"],
        }
    return out


def _mean(xs: list) -> Optional[float]:
    return mean(xs) if xs else None
