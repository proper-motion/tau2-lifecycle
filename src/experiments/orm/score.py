"""Score the ORM against the LLM-judge arms and the baselines (R16, D16, D17).

Primary rule (preregistered): on the test split's support calls (calls a
customer started because of a problem, not to cancel or change plan), AUROC(ORM)
minus AUROC(best judge arm, chosen on validation support calls) for
churn_or_downgrade_28d, with a paired bootstrap over customers. All calls are
reported as secondary. CI lower bound > 0: ORM wins. Upper bound < 0: judge
wins. Otherwise inconclusive. Everything else is reported, never decisive.

    python -m experiments.orm.score --split runs/x/orm --private runs/x/export/private/eval_private.jsonl \
        --preds orm=preds/orm_test.jsonl judge_zero=... judge_few=... \
        --val-preds judge_zero=preds/judge_zero_val.jsonl judge_few=... --out runs/x/orm/score.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.orm.baselines import (
    observable_predictions,
    random_predictions,
    task_success_predictions,
)
from experiments.orm.common import (
    auroc,
    brier,
    cluster_bootstrap,
    read_jsonl,
    spearman,
    top_k_capture,
)

UNINFORMATIVE = {"p_churn": 0.5, "p_repeat": 0.5, "csat_pred": 3.0}
JUDGE_ARMS = ("judge_zero", "judge_few")


def align(rows: list[dict], preds: list[dict]) -> tuple[dict[str, np.ndarray], int]:
    """Predictions in row order; missing values become uninformative and are counted."""
    by_id = {p["row_id"]: p for p in preds}
    missing = 0
    out = {k: [] for k in UNINFORMATIVE}
    for r in rows:
        p = by_id.get(r["row_id"], {})
        for k, default in UNINFORMATIVE.items():
            v = p.get(k)
            if v is None:
                missing += k == "p_churn"
                v = default
            out[k].append(float(v))
    return {k: np.array(v) for k, v in out.items()}, missing


def arm_table(rows, preds_by_arm, private, seed) -> dict:
    y = np.array([r["churn_or_downgrade_28d"] for r in rows], float)
    yr = np.array([r["repeat_contact_7d"] for r in rows], float)
    cs = np.array([np.nan if r["csat"] is None else r["csat"] for r in rows], float)
    harm = np.array(
        [
            private.get(r["row_id"], {}).get("direct_hazard_contribution") or 0.0
            for r in rows
        ]
    )
    cust = np.array([r["customer_key"] for r in rows])
    table = {}
    for arm, preds in preds_by_arm.items():
        p, missing = align(rows, preds)
        point, lo, hi = cluster_bootstrap(
            cust, lambda i: auroc(y[i], p["p_churn"][i]), seed=seed
        )
        rated = ~np.isnan(cs)
        table[arm] = {
            "churn_auroc": point,
            "churn_auroc_ci": [lo, hi],
            "churn_brier": brier(y, p["p_churn"]),
            "repeat_auroc": auroc(yr, p["p_repeat"]),
            "csat_spearman": spearman(cs[rated], p["csat_pred"][rated])
            if rated.sum() > 2
            else None,
            "top10_harm_capture": top_k_capture(harm, p["p_churn"]),
            "missing": missing,
        }
    return table


def primary(rows, orm_preds, judge_preds, seed) -> dict:
    y = np.array([r["churn_or_downgrade_28d"] for r in rows], float)
    cust = np.array([r["customer_key"] for r in rows])
    po, _ = align(rows, orm_preds)
    pj, _ = align(rows, judge_preds)
    diff, lo, hi = cluster_bootstrap(
        cust,
        lambda i: auroc(y[i], po["p_churn"][i]) - auroc(y[i], pj["p_churn"][i]),
        seed=seed,
    )
    verdict = "orm_wins" if lo > 0 else "judge_wins" if hi < 0 else "inconclusive"
    return {
        "auroc_diff": diff,
        "ci95": [lo, hi],
        "verdict": verdict,
        "n_rows": len(rows),
        "n_customers": int(len(set(cust))),
        "n_positive": int(y.sum()),
    }


def support_only(rows: list[dict], private: dict[str, dict]) -> list[dict]:
    """Calls with no cancel/downgrade intent (the call type is private, scoring only)."""
    return [r for r in rows if private.get(r["row_id"], {}).get("intent") is None]


def best_judge_on_val(val_rows, val_preds: dict[str, list[dict]]) -> str:
    y = np.array([r["churn_or_downgrade_28d"] for r in val_rows], float)
    scores = {
        arm: auroc(y, align(val_rows, p)[0]["p_churn"]) for arm, p in val_preds.items()
    }
    return max(scores, key=lambda a: -np.inf if np.isnan(scores[a]) else scores[a])


def run(
    split: Path,
    private_path: Path,
    preds: dict,
    val_preds: dict,
    seed: int = 13,
    heldout: Path | None = None,
    heldout_preds: dict | None = None,
) -> dict:
    train, val, test = (
        read_jsonl(split / f"{n}.jsonl") for n in ("train", "val", "test")
    )
    private = {r["row_id"]: r for r in read_jsonl(private_path)}
    load = lambda d: {arm: read_jsonl(p) for arm, p in d.items()}  # noqa: E731
    preds, val_preds = load(preds), load(val_preds)
    best = best_judge_on_val(support_only(val, private), val_preds)
    support = support_only(test, private)
    arms = {
        **preds,
        "observable_logistic": observable_predictions(train, test),
        "random": random_predictions(test, seed),
        "task_success_ref": task_success_predictions(test, private),
    }
    report = {
        "primary": {
            "scope": "support calls",
            "best_judge_arm": best,
            **primary(support, preds["orm"], preds[best], seed),
        },
        "secondary_all_calls": {
            "best_judge_arm": best,
            **primary(test, preds["orm"], preds[best], seed),
        },
        "in_distribution_support": arm_table(support, arms, private, seed),
        "in_distribution_all": arm_table(test, arms, private, seed),
    }
    if heldout is not None:
        ho = read_jsonl(heldout / "test.jsonl")
        report["heldout"] = arm_table(ho, load(heldout_preds or {}), private, seed)
    return report


def _pairs(items: list[str]) -> dict[str, Path]:
    return {k: Path(v) for k, v in (x.split("=", 1) for x in items)}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--private", type=Path, required=True)
    ap.add_argument(
        "--preds", nargs="+", required=True, help="arm=path; needs orm and judge arms"
    )
    ap.add_argument(
        "--val-preds", nargs="+", required=True, help="judge arm=path on val"
    )
    ap.add_argument("--heldout", type=Path)
    ap.add_argument("--heldout-preds", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    report = run(
        args.split,
        args.private,
        _pairs(args.preds),
        _pairs(args.val_preds),
        args.seed,
        args.heldout,
        _pairs(args.heldout_preds),
    )
    args.out.write_text(json.dumps(report, indent=2))
    p = report["primary"]
    print(
        f"{p['verdict']}: AUROC diff {p['auroc_diff']:.3f} (95% CI {p['ci95'][0]:.3f} to {p['ci95'][1]:.3f}) "
        f"vs {p['best_judge_arm']}, {p['n_customers']} customers, {p['n_positive']} positives"
    )
    return report


if __name__ == "__main__":
    main()
