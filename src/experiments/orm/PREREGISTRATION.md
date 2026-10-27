# Preregistration: outcome reward model vs LLM-as-judge

Status: **DRAFT**. Fill every `[FILL]` and commit before the pilot runs. After the
pilot gate passes, the hazard config hash and full-run size are added and the file
is frozen (git tag `prereg-v1`) before the full run starts. Later edits are listed
under Deviations, never silently.

## Claim

Within this simulator family (τ²-bench telecom + the authored lifecycle hazard),
does a fine-tuned small outcome reward model predict a customer's 28-day churn or
downgrade from a single support transcript better than a frontier LLM judge?
This is a prediction claim. "Direct hazard contribution" is the authored model's
per-call term, not a causal effect, and no result is described as retention skill
or real-customer behavior.

## Data

- Campaign: `tau2 lifecycle run`, config hash `[FILL after pilot]`, 9 variants =
  3 agent models `[FILL]` × tool-failure rate {0, 0.1, 0.3}.
- Rows: `tau2 lifecycle export` at the final cutoff. Only mature, non-infra
  conversations. Fields: transcript + labels (`churn_or_downgrade_28d`,
  `repeat_contact_7d`, `csat`). No variant, task, intent, reward or oracle field.
- Split: `build_dataset.py --seed 13`, by customer: 70/15/15 train/val/test. All
  conversations and variants of a customer share a split.
- Held-out set (D16): a second campaign with 2 perturbed hazard configs (weights
  ±30%, one extra hidden trait) and 1 agent model not in training `[FILL]`. Same
  population size as the test split.

## Arms

| Arm | Input | Notes |
|---|---|---|
| `orm` | transcript | `train_orm.py`: `[FILL base model]`, LoRA r=16, 3 heads, best val loss |
| `judge_zero` | transcript | `judge_baseline.py --arm zero`, model `[FILL]`, family ≠ CSAT rater |
| `judge_few` | transcript + 8 train examples | stratified by churn label, seed 13 |
| `observable_logistic` | transcript counts | reference |
| `random` | none | floor |
| `task_success_ref` | private native reward | reference only, not a competitor |

The best judge arm is the one with the higher validation churn AUROC.

## Primary rule (R16)

On the test split: Δ = AUROC(`orm`) − AUROC(best judge arm) for
`churn_or_downgrade_28d`. 95% CI from a paired bootstrap over customers
(2,000 resamples, seed 13).

- CI lower bound > 0 → **ORM wins**
- CI upper bound < 0 → **judge wins**
- otherwise → **inconclusive**

Missing judge predictions (after 3 attempts) are scored as uninformative (0.5)
and their count is reported.

## Secondary (reported, never decisive)

Per arm: churn AUROC with CI, Brier, repeat-contact AUROC, CSAT Spearman, top-10%
capture of direct hazard contribution. The same table on the held-out set.

## Sizing (D17)

The full run must contain at least **200 distinct churn-or-downgrade events in the
test split**, counted once per branch however many conversation windows each
labels. Size = pilot event rate × 1.3 margin. Budget cap = measured cost per
conversation × planned conversations × 1.3.

## Pilot gate (D15, D18)

`pilot_gate.py`, pilot = 30 customers × 9 variants × 60 days, cap $75, at most 2
re-pilots (each ≤ $75). All must pass before the full run:

- churn-or-downgrade per branch-month in [3%, 10%]
- 7-day repeat contact in [10%, 30%]
- transcript-only logistic AUROC, 95% CI lower bound > 0.55
- best-minus-worst variant churn difference, 95% CI excludes 0

Pilot data is never used in the experiment. Hazard weights may change only
between pilots and are frozen after the passing pilot.

## Deviations

_(none yet)_
