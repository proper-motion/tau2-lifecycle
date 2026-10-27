# ORM vs LLM-as-judge experiment

Does a small fine-tuned outcome reward model predict churn from a support
transcript better than a frontier LLM judge? Rules: `PREREGISTRATION.md`.

```bash
# 1. pilot, then gate (see tau2.lifecycle README for campaigns)
doppler run -- uv run tau2 lifecycle run --config pilot.yaml --out runs/pilot
uv run python -m experiments.orm.pilot_gate --run runs/pilot --cutoff 60

# 2. full run + export + split
doppler run -- uv run tau2 lifecycle run --config full.yaml --out runs/full
uv run tau2 lifecycle export --out runs/full --cutoff <final day>
uv run python -m experiments.orm.build_dataset --export runs/full/export --out runs/full/orm --seed 13

# 3. predictions
doppler run -- uv run python -m experiments.orm.judge_baseline --split runs/full/orm --part val \
    --arm few --model <judge> --csat-model <csat rater> --out runs/full/preds/judge_few_val.jsonl
#    (repeat for zero/few × val/test)
python -m experiments.orm.train_orm --split runs/full/orm --base-model <hf id> \
    --max-tokens <p95> --out runs/full/model --predict test   # GPU box, requirements.txt

# 4. verdict
uv run python -m experiments.orm.score --split runs/full/orm \
    --private runs/full/export/private/eval_private.jsonl \
    --preds orm=runs/full/model/preds_test.jsonl judge_zero=... judge_few=... \
    --val-preds judge_zero=... judge_few=... --out runs/full/score.json
```

Tests (offline, toy data): `uv run pytest tests/orm -q`.
