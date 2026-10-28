# Preregistration: outcome reward model vs LLM-as-judge

Status: **DRAFT**. Model choices filled 2026-10-27. After the pilot gate passes,
the config hash (the one `[FILL]`) and full-run size are added and the file is
frozen (git tag `prereg-v1`) before the full run starts. Later edits are listed
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
  3 agent models × tool-failure rate {0, 0.1, 0.3}. Agent models, chosen for a
  spread in published τ²-bench telecom pass^1 (repo leaderboard data,
  `web/leaderboard/public/submissions`, checked 2026-10-27):
  - weak: `gpt-4.1-2025-04-14` (34.0)
  - mid: `gpt-4.1-mini-2025-04-14` (48.9)
  - strong: `gpt-5-2025-08-07` (95.8)
- Simulated customer: `gpt-4.1-2025-04-14` (upstream default user model), the
  same for every variant. CSAT rater: `gpt-4.1-mini-2025-04-14`.
- All models pinned to dated snapshots so the run can be repeated.
- Rows: `tau2 lifecycle export` at the final cutoff. Only mature, non-infra
  conversations. Fields: transcript + labels (`churn_or_downgrade_28d`,
  `repeat_contact_7d`, `csat`). No variant, task, intent, reward or oracle field.
- Split: `build_dataset.py --seed 13`, by customer: 70/15/15 train/val/test. All
  conversations and variants of a customer share a split.
- Held-out set (D16): a second campaign with 2 perturbed hazard configs (weights
  ±30%, one extra hidden trait) and 1 agent model not in training:
  `o4-mini-2025-04-16` (published telecom pass^1 50.2, mid strength). Same
  population size as the test split.

## Arms

| Arm | Input | Notes |
|---|---|---|
| `orm` | transcript | `train_orm.py`: `Qwen/Qwen3-4B` @ `1cfa9a72`, `--max-tokens 8192`, LoRA r=16, 3 heads, best val loss |
| `judge_zero` | transcript | `judge_baseline.py --arm zero --allow-same-family`, model `gpt-5.5-2026-04-23` |
| `judge_few` | transcript + 8 train examples | stratified by churn label, seed 13 |
| `observable_logistic` | transcript counts | reference |
| `random` | none | floor |
| `task_success_ref` | private native reward | reference only, not a competitor |

The best judge arm is the one with the higher validation churn AUROC. Both judge
arms use `gpt-5.5-2026-04-23` at temperature 0.

ORM base model rationale: text-only causal LM with 41k context, Apache-2.0,
works with `AutoModel` in `train_orm.py`. Pilot transcripts are about 1.4k-2.7k
tokens (live smoke, 2026-10-27), so 8,192 tokens covers them; confirm against
the pilot's measured p95 before freezing. `Qwen/Qwen3.5-4B` was not chosen: it
is an image-text model (`Qwen3_5ForConditionalGeneration`).

## Primary rule (R16)

On the test split's **support calls** (no cancel or downgrade intent):
Δ = AUROC(`orm`) − AUROC(best judge arm) for `churn_or_downgrade_28d`. The best
judge arm is chosen on validation support calls. The same Δ on all calls is
reported as secondary. 95% CI from a paired bootstrap over customers
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

`pilot_gate.py`, pilot = 30 customers × 9 variants × 60 days, cap $110
(deviation 2), at most 2 re-pilots (each ≤ $110). All must pass before the full run:

- churn-or-downgrade per branch-month in [3%, 10%]
- 7-day repeat contact in [10%, 30%]
- transcript-only logistic AUROC, 95% CI lower bound > 0.55
- best-minus-worst variant churn difference, 95% CI excludes 0

Pilot data is never used in the experiment. Hazard weights may change only
between pilots and are frozen after the passing pilot.

## Deviations

1. **Judge and CSAT rater share a model family (OpenAI)**, against D14. Decided
   by Adi on 2026-10-27 to avoid a second provider for now. Risk: the judge may
   share the CSAT rater's biases, which favours the judge on CSAT and, through the
   hazard's CSAT term, slightly on churn. Report this next to any judge result;
   rerun with a non-OpenAI judge before any external claim.
2. **Pilot cap raised from $75 to $110**, against D15. Decided by Adi on
   2026-10-27 after a live check: one gpt-5 agent conversation cost $0.255
   (77 turns), putting the ~800-call pilot at about $85-95. Under $75 the budget
   guard would stop around day 50, before late churn windows mature. Models
   unchanged. Re-pilots are capped at $110 each.
3. **Pilot 1 accepted although the gate failed 3 of 4 checks**, against D15/D18.
   Decided by Adi on 2026-10-28; no re-pilot. Pilot 1: 60 days, 716 calls,
   $96.19. Results:
   - churn 10.2% per branch-month vs a 10% ceiling
   - 7-day repeat contact 41% vs a 30% ceiling
   - transcript-count logistic AUROC 0.54, CI [0.40, 0.68]
   - variant difference passed

   Reasons: the two band misses are realism and cost settings, not threats to
   the ORM-vs-judge comparison. The signal check measured the count features,
   not the data. Churn follows task failure (53% vs 14%, AUROC 0.73, customer CI
   [0.65, 0.80]), so transcripts that reveal resolution carry learnable signal.
4. **Injected tool failures rotate over 8 realistic error messages** instead of
   one fixed string, so a trained model cannot win by memorising a marker.
   Applied before the full run.
5. **Primary verdict scored on support calls only** (all calls secondary). Decided
   by Adi on 2026-10-28 before the full run. In pilot 1, cancel/downgrade calls
   were 12% of calls but 41% of churn labels (69% churn vs 23% for support calls):
   the customer announces the outcome, so both predictors score them easily and
   they shrink the measured gap. The call type comes from the private file and
   is used only to select rows for scoring.
