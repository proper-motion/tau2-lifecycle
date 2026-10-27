"""LLM-as-judge baseline: zero-shot and 8-shot arms (D14).

The judge reads exactly the training-row transcript the ORM reads and outputs
the same three predictions. Few-shot examples come from the train split only.
The judge must be a different model family from the CSAT rater, which wrote
part of the CSAT target.

    python -m experiments.orm.judge_baseline --split runs/x/orm --part test \
        --arm few --model claude-sonnet-... --csat-model gpt-4.1-mini --out preds/judge_few_test.jsonl
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from experiments.orm.common import read_jsonl, write_jsonl

K_SHOTS = 8
MAX_TRANSCRIPT_CHARS = 12_000
EXAMPLE_TRANSCRIPT_CHARS = 2_500

RUBRIC = """You analyse customer-support conversations for a telecom company.
From the transcript alone, predict what this customer does next:
- p_churn: probability (0-1) they cancel their line or downgrade their plan within 28 days
- p_repeat: probability (0-1) they contact support again within 7 days
- csat_pred: the satisfaction rating (1-5) they give this conversation
Answer with JSON only: {"p_churn": <float>, "p_repeat": <float>, "csat_pred": <float>}"""


def family(model: str) -> str:
    m = model.lower().split("/")[-1]
    for prefix, fam in (
        ("gpt", "openai"),
        ("o1", "openai"),
        ("o3", "openai"),
        ("o4", "openai"),
        ("claude", "anthropic"),
        ("gemini", "google"),
        ("llama", "meta"),
    ):
        if m.startswith(prefix):
            return fam
    return m.split("-")[0]


def tail(text: str, n: int) -> str:
    """Keep the end of long transcripts: outcomes are decided late in the call."""
    return text if len(text) <= n else "[...]\n" + text[-n:]


def pick_shots(train: list[dict], seed: int, k: int = K_SHOTS) -> list[dict]:
    """k examples from train only, half churn-positive when possible."""
    rng = random.Random(seed)
    pos = [r for r in train if r["churn_or_downgrade_28d"] == 1]
    neg = [r for r in train if r["churn_or_downgrade_28d"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = min(len(pos), k // 2)
    shots = pos[:n_pos] + neg[: k - n_pos]
    rng.shuffle(shots)
    return shots


def build_messages(row: dict, shots: list[dict]) -> list[dict]:
    content = RUBRIC
    for s in shots:
        label = {
            "p_churn": s["churn_or_downgrade_28d"],
            "p_repeat": s["repeat_contact_7d"],
            "csat_pred": s["csat"],
        }
        content += f"\n\nExample transcript:\n{tail(s['transcript'], EXAMPLE_TRANSCRIPT_CHARS)}\nOutcome: {json.dumps(label)}"
    content += f"\n\nTranscript:\n{tail(row['transcript'], MAX_TRANSCRIPT_CHARS)}"
    return [{"role": "user", "content": content}]


def parse(text: str) -> Optional[dict]:
    try:
        d = json.loads(text[text.index("{") : text.rindex("}") + 1])
        out = {
            "p_churn": float(d["p_churn"]),
            "p_repeat": float(d["p_repeat"]),
            "csat_pred": float(d["csat_pred"]),
        }
    except (ValueError, KeyError, TypeError):
        return None
    ok = (
        0 <= out["p_churn"] <= 1
        and 0 <= out["p_repeat"] <= 1
        and 1 <= out["csat_pred"] <= 5
    )
    return out if ok else None


def litellm_call(model: str) -> Callable[[list[dict]], str]:
    import litellm

    def call(messages: list[dict]) -> str:
        resp = litellm.completion(model=model, messages=messages, temperature=0)
        return resp.choices[0].message.content

    return call


def predict(
    rows: list[dict], shots: list[dict], call, attempts: int = 3, workers: int = 8
) -> list[dict]:
    """Missing predictions are recorded as None and scored as uninformative."""

    def one(row):
        for _ in range(attempts):
            try:
                pred = parse(call(build_messages(row, shots)))
            except Exception:
                pred = None
            if pred:
                return {"row_id": row["row_id"], **pred}
        return {
            "row_id": row["row_id"],
            "p_churn": None,
            "p_repeat": None,
            "csat_pred": None,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, rows))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split", type=Path, required=True, help="Dir with train/val/test.jsonl"
    )
    ap.add_argument("--part", choices=["val", "test", "heldout"], required=True)
    ap.add_argument("--arm", choices=["zero", "few"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--csat-model", required=True, help="Model that rated CSAT in the campaign"
    )
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--allow-same-family",
        action="store_true",
        help="Accepted deviation from D14; must be listed in PREREGISTRATION.md",
    )
    args = ap.parse_args(argv)
    if family(args.model) == family(args.csat_model) and not args.allow_same_family:
        raise SystemExit(
            f"Judge {args.model} and CSAT rater {args.csat_model} are both "
            f"{family(args.model)}; D14 requires different families."
        )
    shots = (
        pick_shots(read_jsonl(args.split / "train.jsonl"), args.seed)
        if args.arm == "few"
        else []
    )
    rows = read_jsonl(args.split / f"{args.part}.jsonl")
    preds = predict(rows, shots, litellm_call(args.model))
    write_jsonl(args.out, preds)
    print({"rows": len(rows), "missing": sum(p["p_churn"] is None for p in preds)})


if __name__ == "__main__":
    main()
