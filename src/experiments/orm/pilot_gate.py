"""Pilot gate (D15 + D18): measure the pilot, then decide whether the full run may start.

Checks:
  cost and trace length    reported (they size the full run and pick the ORM model)
  churn_band               churn-or-downgrade per branch-month in [3%, 10%]
  repeat_band              7-day repeat contact on mature labels in [10%, 30%]
  signal_observable        transcript-only logistic AUROC, 95% CI lower bound > 0.55;
                           features = transcript counts + an LLM "was the issue
                           resolved?" reading of the transcript (deviation 3)
  agent_matters            churn-or-downgrade rate, best minus worst variant, 95% CI excludes 0

    python -m experiments.orm.pilot_gate --run runs/pilot --cutoff 60 --resolution-model gpt-4.1-mini
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from experiments.orm.baselines import Logistic, observable_features
from experiments.orm.build_dataset import assign_split
from experiments.orm.common import auroc, cluster_bootstrap

CHURN_BAND = (0.03, 0.10)
REPEAT_BAND = (0.10, 0.30)
SIGNAL_FLOOR = 0.55
RESOLUTION_PROMPT = (
    "Below is a customer-support transcript. Judging only from the transcript, was "
    "the customer's problem fully resolved by the end of the call? Answer with one "
    "word: yes or no.\n\n"
)


def resolution_scores(
    rows: list[dict], call: Callable[[str], str], cache: Optional[Path] = None
) -> dict[str, float]:
    """1.0 resolved, 0.0 not, 0.5 unreadable. Cached by row_id so reruns cost nothing."""
    done = {}
    if cache and cache.exists():
        done = {
            d["row_id"]: d["resolved"]
            for d in map(json.loads, cache.read_text().splitlines())
        }

    def one(r):
        try:
            ans = call(RESOLUTION_PROMPT + r["transcript"][-12_000:]).strip().lower()
        except Exception:
            return r["row_id"], 0.5
        return r["row_id"], 1.0 if ans.startswith("yes") else 0.0 if ans.startswith(
            "no"
        ) else 0.5

    todo = [r for r in rows if r["row_id"] not in done]
    with ThreadPoolExecutor(max_workers=8) as pool:
        new = dict(pool.map(one, todo))
    if cache and new:
        with cache.open("a") as f:
            for rid, v in new.items():
                f.write(json.dumps({"row_id": rid, "resolved": v}) + "\n")
    return {**done, **new}


def litellm_reader(model: str) -> Callable[[str], str]:
    import litellm

    def call(prompt: str) -> str:
        resp = litellm.completion(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return resp.choices[0].message.content

    return call


def gate(
    conn, cutoff_day: int, seed: int = 13, resolved: Optional[dict[str, float]] = None
) -> dict:
    """`resolved` maps row_id -> transcript-only resolution reading (deviation 3)."""
    from tau2.lifecycle.export import private_eval_rows, training_rows

    rows = training_rows(conn, cutoff_day)
    private = {r["row_id"]: r for r in private_eval_rows(conn, cutoff_day)}
    convs = [
        json.loads(p)
        for (p,) in conn.execute("SELECT payload FROM events WHERE type='conversation'")
    ]
    costs = [c["cost_usd"] for c in convs if not c["infra_error"]]
    lengths = [len(c["transcript"]) / 4 for c in convs if not c["infra_error"]]

    # Per-branch churn-or-downgrade by cutoff (customer-level, D17)
    branches = [b for (b,) in conn.execute("SELECT DISTINCT branch_id FROM events")]
    left = {
        b
        for (b,) in conn.execute(
            "SELECT DISTINCT branch_id FROM events WHERE virtual_day <= ? AND (type='churned'"
            " OR (type='plan_changed' AND json_extract(payload, '$.direction')='down'))",
            (cutoff_day,),
        )
    }
    months = max(cutoff_day / 30, 1e-9)
    churn_monthly = len(left) / max(len(branches), 1) / months
    repeat = [
        r["repeat_contact_7d"] for r in rows if r["repeat_contact_7d"] is not None
    ]
    repeat_rate = float(np.mean(repeat)) if repeat else float("nan")

    # Signal check: 2-fold cross-fit by customer so no row is scored in-sample
    y = np.array([r["churn_or_downgrade_28d"] for r in rows], float)
    X = (
        np.array(
            [
                np.r_[
                    observable_features(r["transcript"]),
                    [] if resolved is None else [resolved.get(r["row_id"], 0.5)],
                ]
                for r in rows
            ]
        )
        if rows
        else np.zeros((0, 6))
    )
    fold = np.array([assign_split(r["customer_key"], seed) == "train" for r in rows])
    p = np.full(len(rows), 0.5)
    for f in (True, False):
        tr, te = fold == f, fold != f
        if te.any() and tr.any() and 0 < y[tr].mean() < 1:
            p[te] = Logistic().fit(X[tr], y[tr]).predict_proba(X[te])
    cust = np.array([r["customer_key"] for r in rows])
    sig = (
        cluster_bootstrap(cust, lambda i: auroc(y[i], p[i]), seed=seed)
        if rows
        else (np.nan,) * 3
    )

    # Agent matters: paired over population customers
    by_variant: dict[str, dict[str, int]] = {}
    for b in branches:
        pop, v = b.split(":", 1)
        by_variant.setdefault(v, {})[pop] = int(b in left)
    rates = {v: np.mean(list(d.values())) for v, d in by_variant.items()}
    best, worst = min(rates, key=rates.get), max(rates, key=rates.get)
    pops = np.array(sorted(by_variant[best]))
    a = np.array([by_variant[worst].get(x, 0) for x in pops])
    b_ = np.array([by_variant[best].get(x, 0) for x in pops])
    diff = cluster_bootstrap(
        pops, lambda i: float(a[i].mean() - b_[i].mean()), seed=seed
    )

    checks = {
        "churn_band": {
            "value": churn_monthly,
            "band": CHURN_BAND,
            "pass": CHURN_BAND[0] <= churn_monthly <= CHURN_BAND[1],
        },
        "repeat_band": {
            "value": repeat_rate,
            "band": REPEAT_BAND,
            "pass": REPEAT_BAND[0] <= repeat_rate <= REPEAT_BAND[1],
        },
        "signal_observable": {
            "auroc": sig[0],
            "ci95": [sig[1], sig[2]],
            "pass": bool(sig[1] > SIGNAL_FLOOR),
        },
        "agent_matters": {
            "worst": worst,
            "best": best,
            "diff": diff[0],
            "ci95": [diff[1], diff[2]],
            "pass": bool(diff[1] > 0 or diff[2] < 0),
        },
    }
    return {
        "pass": all(c["pass"] for c in checks.values()),
        "checks": checks,
        "measured": {
            "conversations": len(convs),
            "cost_per_conversation_usd": float(np.mean(costs)) if costs else None,
            "p95_transcript_tokens": float(np.percentile(lengths, 95))
            if lengths
            else None,
            "training_rows": len(rows),
            "injected_rows": sum(1 for r in private.values() if r["injected_failures"]),
        },
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="Campaign --out directory")
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--resolution-model", help="LLM that reads transcripts for resolution"
    )
    args = ap.parse_args(argv)
    from tau2.lifecycle.export import training_rows
    from tau2.lifecycle.world import open_ledger

    conn = open_ledger(args.run / "ledger.sqlite")
    resolved = None
    if args.resolution_model:
        resolved = resolution_scores(
            training_rows(conn, args.cutoff),
            litellm_reader(args.resolution_model),
            cache=args.run / "gate_resolution.jsonl",
        )
    report = gate(conn, args.cutoff, args.seed, resolved)
    print(json.dumps(report, indent=2, default=float))
    print("PASS" if report["pass"] else "FAIL")
    return report


if __name__ == "__main__":
    main()
