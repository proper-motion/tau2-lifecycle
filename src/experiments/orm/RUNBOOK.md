# Runbook: full run for ORM vs LLM-as-judge

For whoever runs the full experiment. Rules live in `PREREGISTRATION.md`,
unit costs in `COSTS.md`. Everything here was measured on pilot 1 (2026-10-28).

## 1. Pick a size

The only knob is **`customers`** in `configs/full.yaml`. Everything else stays as
preregistered (9 variants, 60 days, same models, seed 7).

What decides the verdict is how many **churned support calls land in the test
split** (15% of customers; primary score = support calls only, deviation 5),
spread over how many test customers.

| Size | Customers | Test customers | Churned support calls in test (≈) | API cost (≈) | Wall-clock at concurrency 16 (≈) | What it can show |
|---|---|---|---|---|---|---|
| Smoke | 50 | ~8 | ~20 | ~$160 | ~4 h | Pipeline works end to end; no verdict |
| Demo | 250 | ~38 | ~100 | ~$800 | ~8 h | ORM vs judge only if the gap is large (≈0.10 AUROC) |
| Solid | 500 | ~75 | ~200 | ~$1,600 | ~14 h | Medium gaps (≈0.07 AUROC); good default |
| Strong | 1,000 | ~150 | ~400 | ~$3,200 | ~26 h | Small gaps (≈0.05 AUROC); external claims |

Pilot rates behind the table: $3.21 per customer (all 9 variants, 60 days); 2.7
churned support-call rows per customer over the mature half of the horizon.
Gap sizes are rough rules of thumb, not a power analysis.

**Recommendation:** run **Smoke** first. It is cheap and proves the full chain
(campaign → export → split → judge → ORM training → score). Then run **Solid**.
Pick **Strong** only if Solid comes back inconclusive with a gap pointing toward
the ORM.

Optional: the held-out simulation (D16) adds ~15-20% and answers a different
question ("does the ORM survive changed churn rules?"). It is not needed for the
v1 claim.

## 2. Run it

```bash
cd tau2-lifecycle
doppler setup --no-interactive            # once; keys come from Doppler gaia/prd
# edit configs/full.yaml: set customers and a new name
doppler run -- uv run tau2 lifecycle run \
    --config src/experiments/orm/configs/full.yaml --out runs/full
```

- **Keep the machine awake with the lid open.** Lid-close sleep pauses the run. A
  cloud VM or desktop is better for the 8-26 h sizes.
- **Interrupted?** Re-run the same command; it resumes at the last finished day
  and never re-pays a recorded call. A changed config is refused, by design.
- **Stop early:** `uv run tau2 lifecycle cancel --out runs/full`.
- **Spend:** the hard cap is `budget_usd` in the config. Set it to about 1.2× the
  table's cost.

## 3. After the run

```bash
uv run tau2 lifecycle export --out runs/full --cutoff 59
uv run python -m experiments.orm.build_dataset --export runs/full/export --out runs/full/orm --seed 13
```

Then follow `README.md` steps 3-4: run the judge arms (val + test), train the ORM
on a rented GPU (`train_orm.py`, about $10-20), and run `score.py` for the verdict.
Before the run starts, freeze `PREREGISTRATION.md` (add the config hash and
size, `git tag prereg-v1`).
