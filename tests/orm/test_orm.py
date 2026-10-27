"""Phase 6: ORM experiment harness on toy data (no LLM, no GPU)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.orm import baselines, build_dataset, judge_baseline, score, train_orm
from experiments.orm.common import (
    auroc,
    cluster_bootstrap,
    spearman,
    top_k_capture,
    write_jsonl,
)


def toy_rows(n_customers=60, per_customer=4, seed=0):
    """Churn is driven by tool errors in the transcript, plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        for k in range(per_customer):
            errors = int(rng.integers(0, 4))
            churn = int(rng.random() < 0.1 + 0.25 * errors)
            transcript = (
                "assistant: hi\nuser: help\n"
                + "assistant -> refuel_data({})\ntool: Error: timeout\n" * errors
            )
            rows.append(
                {
                    "row_id": f"r{c}_{k}",
                    "customer_key": f"c{c}",
                    "virtual_day": k,
                    "transcript": transcript,
                    "churn_or_downgrade_28d": churn,
                    "repeat_contact_7d": int(rng.random() < 0.2),
                    "csat": None if k == 0 else 5 - errors,
                }
            )
    return rows


# ---------------------------------------------------------------- metrics


def test_auroc_matches_known_values():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(auroc([1, 1], [0.2, 0.3]))


def test_spearman_and_top_k_capture():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert (
        top_k_capture(
            np.array([0, 0, 10, 0, 0, 0, 0, 0, 0, 0]), np.arange(10)[::-1] == 7
        )
        == 1.0
    )


def test_cluster_bootstrap_resamples_whole_clusters():
    clusters = np.repeat(np.arange(20), 5)
    seen = []
    cluster_bootstrap(clusters, lambda i: seen.append(len(i)) or 0.0, n_boot=50)
    assert all(n % 5 == 0 for n in seen)


# ---------------------------------------------------------------- split


def test_split_keeps_each_customer_in_one_split_and_is_deterministic():
    rows = toy_rows()
    a, b = build_dataset.split_rows(rows, 13), build_dataset.split_rows(rows, 13)
    assert a == b
    owners = {}
    for name, part in a.items():
        for r in part:
            owners.setdefault(r["customer_key"], set()).add(name)
    assert all(len(s) == 1 for s in owners.values())
    assert 0.5 < len(a["train"]) / len(rows) < 0.85


def test_build_dataset_cli_writes_three_files(tmp_path):
    write_jsonl(tmp_path / "export" / "train_rows.jsonl", toy_rows())
    counts = build_dataset.main(
        [
            "--export",
            str(tmp_path / "export"),
            "--out",
            str(tmp_path / "orm"),
            "--seed",
            "13",
        ]
    )
    assert sum(counts.values()) == 240
    assert all(
        (tmp_path / "orm" / f"{n}.jsonl").exists() for n in ("train", "val", "test")
    )


# ---------------------------------------------------------------- baselines


def test_observable_baseline_learns_the_toy_signal():
    rows = toy_rows()
    parts = build_dataset.split_rows(rows, 13)
    preds = baselines.observable_predictions(parts["train"], parts["test"])
    y = np.array([r["churn_or_downgrade_28d"] for r in parts["test"]])
    assert auroc(y, [p["p_churn"] for p in preds]) > 0.7
    assert all(1 <= p["csat_pred"] <= 5 for p in preds)


# ---------------------------------------------------------------- judge


def test_few_shot_examples_come_from_train_only_and_are_stratified():
    parts = build_dataset.split_rows(toy_rows(), 13)
    shots = judge_baseline.pick_shots(parts["train"], seed=13)
    train_ids = {r["row_id"] for r in parts["train"]}
    assert len(shots) == 8 and all(s["row_id"] in train_ids for s in shots)
    assert sum(s["churn_or_downgrade_28d"] for s in shots) == 4
    assert shots == judge_baseline.pick_shots(parts["train"], seed=13)


def test_judge_parses_retries_and_records_missing():
    rows = toy_rows(n_customers=2, per_customer=1)
    replies = iter(
        [
            "not json",
            '{"p_churn": 0.7, "p_repeat": 0.1, "csat_pred": 2}',
            "garbage",
            "x",
            "y",
        ]
    )
    preds = judge_baseline.predict(rows, [], lambda m: next(replies), workers=1)
    assert preds[0]["p_churn"] == 0.7
    assert preds[1]["p_churn"] is None  # three bad replies -> missing, never invented
    assert (
        judge_baseline.parse('{"p_churn": 1.5, "p_repeat": 0, "csat_pred": 3}') is None
    )


def test_judge_refuses_same_family_as_csat_rater(tmp_path):
    assert judge_baseline.family("openai/gpt-4.1") == judge_baseline.family(
        "gpt-4.1-mini"
    )
    assert judge_baseline.family("anthropic/claude-x") != judge_baseline.family(
        "gpt-4.1-mini"
    )
    with pytest.raises(SystemExit, match="different families"):
        judge_baseline.main(
            [
                "--split",
                str(tmp_path),
                "--part",
                "test",
                "--arm",
                "zero",
                "--model",
                "gpt-4.1",
                "--csat-model",
                "gpt-4.1-mini",
                "--out",
                str(tmp_path / "p.jsonl"),
            ]
        )


def test_long_transcripts_keep_the_end():
    assert judge_baseline.tail("a" * 10 + "END", 5).endswith("a" * 2 + "END")
    ex = train_orm.example(
        {
            "row_id": "r",
            "transcript": "x" * 50 + "END",
            "churn_or_downgrade_28d": 1,
            "repeat_contact_7d": None,
            "csat": None,
        },
        max_chars=10,
    )
    assert ex["text"].endswith("END") and len(ex["text"]) == 10
    assert ex["csat_mask"] == 0.0 and ex["y_repeat"] == 0.0


# ---------------------------------------------------------------- score (R16)


def _setup(tmp_path, orm_quality):
    rows = toy_rows(n_customers=120, per_customer=4)
    parts = build_dataset.split_rows(rows, 13)
    for name, part in parts.items():
        write_jsonl(tmp_path / f"{name}.jsonl", part)
    write_jsonl(
        tmp_path / "private.jsonl",
        [
            {
                "row_id": r["row_id"],
                "task_success": 1.0,
                "direct_hazard_contribution": float(r["churn_or_downgrade_28d"]),
            }
            for r in rows
        ],
    )
    rng = np.random.default_rng(1)

    def preds(part, quality):
        return [
            {
                "row_id": r["row_id"],
                "p_churn": float(
                    quality * r["churn_or_downgrade_28d"] + (1 - quality) * rng.random()
                ),
                "p_repeat": 0.2,
                "csat_pred": 3.0,
            }
            for r in part
        ]

    paths = {}
    for arm, q, part in (
        ("orm", orm_quality, "test"),
        ("judge_zero", 0.3, "test"),
        ("judge_few", 0.3, "test"),
        ("judge_zero_val", 0.2, "val"),
        ("judge_few_val", 0.4, "val"),
    ):
        paths[arm] = tmp_path / f"{arm}.jsonl"
        write_jsonl(paths[arm], preds(parts[part], q))
    return paths


@pytest.mark.parametrize(
    "orm_quality,expected",
    [(0.9, "orm_wins"), (0.3, "inconclusive"), (0.0, "judge_wins")],
)
def test_r16_verdict(tmp_path, orm_quality, expected):
    p = _setup(tmp_path, orm_quality)
    report = score.run(
        tmp_path,
        tmp_path / "private.jsonl",
        {"orm": p["orm"], "judge_zero": p["judge_zero"], "judge_few": p["judge_few"]},
        {"judge_zero": p["judge_zero_val"], "judge_few": p["judge_few_val"]},
    )
    assert report["primary"]["best_judge_arm"] == "judge_few"  # higher val AUROC
    assert report["primary"]["verdict"] == expected
    table = report["in_distribution"]
    assert {
        "orm",
        "judge_few",
        "observable_logistic",
        "random",
        "task_success_ref",
    } <= set(table)


def test_missing_predictions_are_uninformative_and_counted():
    rows = toy_rows(n_customers=3, per_customer=1)
    aligned, missing = score.align(
        rows,
        [
            {
                "row_id": rows[0]["row_id"],
                "p_churn": None,
                "p_repeat": None,
                "csat_pred": None,
            }
        ],
    )
    assert missing == 3 and set(aligned["p_churn"]) == {0.5}


# ---------------------------------------------------------------- pilot gate


def test_pilot_gate_reports_every_check_on_an_offline_campaign(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1] / "lifecycle"))
    from test_campaign import TEMPLATES, _cfg, csat, scripted_runner

    from experiments.orm.pilot_gate import gate
    from tau2.lifecycle.campaign import run_campaign
    from tau2.lifecycle.world import open_ledger

    run_campaign(
        _cfg(days=40, customers=8), tmp_path, scripted_runner(), csat, TEMPLATES
    )
    report = gate(open_ledger(tmp_path / "ledger.sqlite"), cutoff_day=39)
    assert set(report["checks"]) == {
        "churn_band",
        "repeat_band",
        "signal_observable",
        "agent_matters",
    }
    assert report["pass"] == all(c["pass"] for c in report["checks"].values())
    assert report["measured"]["conversations"] > 0
    json.dumps(report, default=float)
