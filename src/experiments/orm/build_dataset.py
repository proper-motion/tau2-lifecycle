"""Customer-level train/val/test split of `tau2 lifecycle export` rows (D6, D17).

All rows of one customer (every conversation, every agent variant) land in the
same split, so no customer seen in training is scored in test.

    python -m experiments.orm.build_dataset --export runs/x/export --out runs/x/orm --seed 13
"""

import argparse
import hashlib
from pathlib import Path

from experiments.orm.common import read_jsonl, write_jsonl

SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def assign_split(customer_key: str, seed: int) -> str:
    u = (
        int(hashlib.sha256(f"{seed}:{customer_key}".encode()).hexdigest()[:12], 16)
        / 16**12
    )
    edge = 0.0
    for name, frac in SPLITS:
        edge += frac
        if u < edge:
            return name
    return SPLITS[-1][0]


def split_rows(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    out = {name: [] for name, _ in SPLITS}
    for r in rows:
        out[assign_split(r["customer_key"], seed)].append(r)
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--export", type=Path, required=True, help="Dir with train_rows.jsonl"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--seed", type=int, required=True, help="Fixed in PREREGISTRATION.md"
    )
    args = ap.parse_args(argv)
    splits = split_rows(read_jsonl(args.export / "train_rows.jsonl"), args.seed)
    for name, rows in splits.items():
        write_jsonl(args.out / f"{name}.jsonl", rows)
    counts = {name: len(rows) for name, rows in splits.items()}
    print(counts)
    return counts


if __name__ == "__main__":
    main()
