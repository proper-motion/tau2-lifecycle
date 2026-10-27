"""Outcome reward model: small open decoder + LoRA + 3 heads (D10).

Heads: P(churn or downgrade <=28d), P(repeat contact <=7d), CSAT. Each example
is the training-row transcript, truncated from the start to the model context.
Rows with no CSAT are masked out of the CSAT loss. The checkpoint with the best
validation loss predicts the requested split.

Runs on a rented GPU (see src/experiments/modal/ for the upstream pattern):

    pip install -r src/experiments/orm/requirements.txt
    python -m experiments.orm.train_orm --split runs/x/orm --base-model <hf-id> \
        --max-tokens 8192 --out runs/x/orm/model --predict test

Data preparation below needs no torch, so it is unit-tested locally.
"""

import argparse
import json
import math
from pathlib import Path

from experiments.orm.common import read_jsonl, write_jsonl


def example(row: dict, max_chars: int) -> dict:
    text = row["transcript"]
    if len(text) > max_chars:
        text = text[-max_chars:]  # keep the end of the call
    csat = row["csat"]
    return {
        "row_id": row["row_id"],
        "text": text,
        "y_churn": float(row["churn_or_downgrade_28d"]),
        "y_repeat": float(row["repeat_contact_7d"] or 0),
        "y_csat": 0.0 if csat is None else (csat - 1) / 4,  # scaled to [0, 1]
        "csat_mask": 0.0 if csat is None else 1.0,
    }


def main(argv=None) -> None:  # pragma: no cover - needs a GPU and torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument(
        "--base-model", required=True, help="Picked at Phase 4 by p95 trace length"
    )
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--predict", choices=["val", "test", "heldout"], default="test")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.padding_side = "left"  # last position = last real token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    base = get_peft_model(
        base,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            target_modules="all-linear",
            lora_dropout=0.05,
        ),
    )
    head = torch.nn.Linear(base.config.hidden_size, 3).to(dev)
    base.to(dev)
    max_chars = args.max_tokens * 4

    def batches(rows, shuffle):
        ex = [example(r, max_chars) for r in rows]
        if shuffle:
            g = torch.Generator().manual_seed(args.seed)
            ex = [ex[i] for i in torch.randperm(len(ex), generator=g)]
        for i in range(0, len(ex), args.batch):
            yield ex[i : i + args.batch]

    def forward(b):
        enc = tok(
            [e["text"] for e in b],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_tokens,
        ).to(dev)
        h = base(**enc).last_hidden_state[:, -1, :].float()
        return head(h)

    def loss_of(out, b):
        t = lambda k: torch.tensor([e[k] for e in b], device=dev)  # noqa: E731
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        l_csat = (
            (torch.sigmoid(out[:, 2]) - t("y_csat")) ** 2 * t("csat_mask")
        ).sum() / t("csat_mask").sum().clamp(min=1)
        return bce(out[:, 0], t("y_churn")) + bce(out[:, 1], t("y_repeat")) + l_csat

    train, val = (
        read_jsonl(args.split / "train.jsonl"),
        read_jsonl(args.split / "val.jsonl"),
    )
    params = [p for p in base.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    best = math.inf
    for epoch in range(args.epochs):
        base.train()
        for b in batches(train, shuffle=True):
            opt.zero_grad()
            loss_of(forward(b), b).backward()
            opt.step()
        base.eval()
        with torch.no_grad():
            vl = sum(
                loss_of(forward(b), b).item() * len(b) for b in batches(val, False)
            ) / max(1, len(val))
        if vl < best:
            best = vl
            base.save_pretrained(args.out / "lora")
            torch.save(head.state_dict(), args.out / "head.pt")
        print({"epoch": epoch, "val_loss": round(vl, 4)})

    head.load_state_dict(torch.load(args.out / "head.pt"))
    rows = read_jsonl(args.split / f"{args.predict}.jsonl")
    preds = []
    with torch.no_grad():
        for b in batches(rows, False):
            p = torch.sigmoid(forward(b))
            for e, (c, r, s) in zip(b, p.tolist()):
                preds.append(
                    {
                        "row_id": e["row_id"],
                        "p_churn": c,
                        "p_repeat": r,
                        "csat_pred": 1 + 4 * s,
                    }
                )
    write_jsonl(args.out / f"preds_{args.predict}.jsonl", preds)
    (args.out / "train_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )


if __name__ == "__main__":
    main()
