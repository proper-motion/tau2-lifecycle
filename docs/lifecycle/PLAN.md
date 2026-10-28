> Design plan and review record for the lifecycle simulator and ORM experiment.
> Copied from the author's working notes on 2026-10-28; local paths and private repo names removed.

# Plan: τ-bench customer lifecycle simulator → outcome reward model vs LLM-as-judge

Review target (fixed): the 7-item spec pasted by Adi on 2026-10-27, plus the stated purpose.
Reviewer: /plan-eng-review (gstack 1.91.280.0), Claude Opus 5.5.
Status: UNDER REVIEW. Nothing implemented.

## Purpose (Adi, 2026-10-27, verbatim intent)

Fine-tune an outcome reward model (ORM): a small judge that predicts how an agent
trace will affect business outcomes. Compare it against a plain LLM-as-judge and see
which does better. Modify τ-bench to create the simulation that produces the
training and evaluation data.

## Original plan (unchanged copy)

Starting from a fresh τ-bench clone, add a customer lifecycle simulator and a clean
viewer around the existing agent, tools and task runner.

1. Persistent customers and product state
   Add customer profiles, preferences, subscriptions, usage and support history.
   Preserve the same customer and account state across conversations.
2. Model-driven customer behavior
   Extend the simulated user beyond the conversation. At virtual-time checkpoints,
   let it decide whether to use the product, return for help, rate the experience,
   renew, downgrade or cancel. Validate and execute those decisions.
3. Event recording and KPI calculations
   Record customer actions in a durable event ledger. Calculate CSAT, repeat contact,
   churn, retention and usage from those events. Keep native task success, execution
   cost and errors as separate measures. Handle pending outcomes and incomplete
   observation windows.
4. Execution tracing
   Instrument conversations, tools and model calls with OpenTelemetry-compatible
   tracing into Langfuse. Distinguish agent, simulated-user and lifecycle-model calls.
5. Product analytics export
   Send simulated business events to either Mixpanel or Amplitude, using consistent
   customer/conversation identities and virtual timestamps. Keep local records
   authoritative and synthetic analytics isolated.
6. Simple showcase UI
   Provide a cohort dashboard, searchable chat/tool logs, customer product state,
   lifecycle timelines and side-by-side agent comparisons. Show how interactions
   connect to later outcomes.
7. Repeatable runs and training exports
   Add batch execution, budgets, cancellation/resume, versioned configurations and
   matched customer populations across agent variants. Export permitted traces and
   mature KPI labels separately from private simulator information.

## Final plan (read this first — approved 2026-10-27, awaiting Adi's review)

**What we build:** public fork [proper-motion/tau2-lifecycle](https://github.com/proper-motion/tau2-lifecycle) of `sierra-research/tau2-bench`, built on upstream `main` at `ebf990b` (2026-10-27, 78 commits past v2.0.0; Adi chose main over the v2.0.0 pin and public over private), telecom domain only. Upstream changes come in only by explicit `git pull upstream main`. It simulates persistent telecom customers across many conversations, decides what they do afterwards with an authored hazard model, and exports labeled traces. Then it trains a small ORM and compares it to an LLM judge.

**Experiment in one line:** does a fine-tuned ~1-4B ORM predict 28-day churn-or-downgrade from a trace better than a frontier LLM judge (zero-shot and 8-shot)? Primary rule: AUROC difference with a customer-cluster bootstrap 95% CI (R16).

**Key decisions:** hybrid hazard decides outcomes; the LLM only rates CSAT (D5). Window labels for training, private direct-hazard-contribution for grading (D7). 3 agent models × 3 tool-failure rates = 9 matched variants (D8). Tasks are retargeted per customer and the customer's ledger history is replayed (D11). Day-wave scheduling on upstream `runner/batch.py` (D12). Pilot gate first: ≤$75, with cost, length, label-rate and signal checks (D15, D18). Held-out simulator test (D16). Cluster-aware sizing (D17). Mixpanel/Amplitude export and the full UI are in scope. Langfuse is deferred.

### Files

```
<new repo>/
  src/tau2/lifecycle/
    world.py       customers, ledger (SQLite WAL, single writer), state projection by branch_id,
                   task retargeting + history replay (D11)
    checkpoint.py  CSAT LLM call, authored hazard (D5), action validation + execution, oracle rows
    kpis.py        window_labels() (single definition), CSAT, repeat contact, churn, retention, usage, censoring
    campaign.py    versioned config + hash, day waves (D12), 1 conversation/branch/day, budget admission,
                   cancel/resume (atomic post-processing), keyed RNG (CRN), tool-failure injection (D8)
    export.py      allowlist training rows (R5), Mixpanel/Amplitude batched sink (item 5)
    cli.py         `tau2 lifecycle run|resume|cancel|export|serve`
  web/
    server.py      FastAPI, 127.0.0.1, read-only ledger, FTS5 search
    index.html, app.js   cohort dashboard, search, customer state, timelines, side-by-side compare
  experiments/orm/
    PREREGISTRATION.md, build_dataset.py, judge_baseline.py, train_orm.py, score.py, pilot_gate.py
  tests/lifecycle/  (see Test Plan artifact)
```

### Phases (one at a time; each ends green before the next)

| Phase | Work | Verify |
|---|---|---|
| 0 | Fork at `v2.0.0` into a new private repo; `uv sync`; record upstream test baseline | `uv run pytest -q` green (upstream suite unmodified) |
| 1 | `world.py`: ledger schema with `branch_id`, projection, synthetic customers, retargeting + history replay, per-conversation applicability check; freeze the ledger schema | `uv run pytest tests/lifecycle -q` incl. `test_retarget_equivalence.py` (D13) |
| 2 | `checkpoint.py` (hazard, CSAT, validation, oracle rows), `kpis.py` (window_labels, censoring) | `uv run pytest tests/lifecycle -q` |
| 3 | `campaign.py` + `cli.py`: day waves, budget admission, cancel/resume, config hash, CRN, injection | offline e2e: `uv run pytest tests/lifecycle/test_campaign_e2e.py -q` (stub models) |
| 4 | `export.py`: allowlist + leakage test, training rows, Mixpanel/Amplitude sink | `uv run pytest tests/lifecycle/test_export.py -q` (sentinel leakage) |
| 5 | `web/`: API + UI over a fixture ledger | `uv run pytest tests/lifecycle/test_web.py -q` + browser smoke of cohort → customer → transcript, search, compare |
| 6 | `experiments/orm/` scaffolding + PREREGISTRATION.md (splits, metrics, arms, R16 rule, D15/D18 thresholds) | `uv run pytest tests/orm -q` on toy data; PREREGISTRATION.md committed before any paid run |
| 7 | Pilot (30 customers × 9 variants × 60 days, ≤$75): measure, tune hazard, apply the D18 signal gate, freeze config | `uv run python experiments/orm/pilot_gate.py --run <pilot>` prints PASS (max 2 re-pilots) |
| 8 | Full run (sized by D17) + held-out campaign (D16); judges + ORM training; scoring | `uv run python experiments/orm/score.py` prints the R16 verdict with CIs |

**Build log:**
- Phase 0 done: 231 focused upstream tests pass.
- Phase 1 done (`proper-motion/tau2-lifecycle`):
  - **D11 simplified:** each branch keeps the template IDs C1001/L1002 inside its own world. Only the name, DOB, email, account status and plan are overlaid through initialization_data.
  - **Applicability check:** adds a phone network re-scan after the gold actions. 108/114 base templates pass upstream; the other 6 (overdue-bill) are excluded.
  - **Known issue for Phase 2:** on the Basic plan only 23/114 templates are applicable, because the template line has 8.7 GB used against a 5 GB cap.
- Tools (`.plans/telecom-lifecycle-tools.md`) done in 3f91b34: telecom_lifecycle domain.
- Phase 2 done in 47effbb:
  - `checkpoint.py`: records each call, classifies intent calls, CSAT (forced 1 when blocked, pending on rater failure), private direct hazard contribution, and a daily hazard with population-keyed draws.
  - `kpis.py`: window labels and cohort KPIs.
  - Default hazard gives about 7% leave intent per 30 days.
  - 37 lifecycle tests pass.
  - The LLM CSAT rater (`llm_csat_rater`) is written but untested live.
- Phase 3 done in f3fe12c: `campaign.py` + `tau2 lifecycle run|cancel|kpis`.
  - **Deviation from D12:** conversations run on upstream `run_simulation` (native grading kept) under our own thread pool. Upstream `run_tasks` requires registered, language-pinned task sets.
  - **Resume:** works through deterministic conversation ids in the ledger, plus a CSAT retry on resume.
  - **Provider outage:** if more than half a day's calls fail, the campaign stops and records nothing for that day.
  - **Infra errors:** support calls that hit one get no CSAT.
  - **Injection:** decided per tool-call id, so evaluator replay matches.
  - **Organic support contacts:** 1 per 30 days by default.
  - **Tests:** 46 lifecycle tests pass offline.
  - **Live path verified** under Doppler (gaia/prd key; the shell key has no credits): 2 real conversations for $0.049 (~$0.024 per call with gpt-4.1-mini agent and user). The CSAT rater works; 1 injected failure was logged.
  - **Watch:** the LLM rater gave CSAT 4 to a failed task. That leniency should be checked in the pilot (D18).
- Phase 4 done: `export.py` + `tau2 lifecycle export|analytics`.
  - **Training export:** built from an allowlist; excludes variant, task id, intent, offer decision, native reward and oracle data.
  - **Private eval file:** a separate file keyed by row id.
  - **Analytics:** Mixpanel/Amplitude senders with retry.
  - **Tests:** sentinel leakage tests pass; 284 tests total.
  - **Not done:** no live analytics send yet, because there are no MIXPANEL/AMPLITUDE keys in Doppler gaia/prd.
- Phase 6 done in 123eb45: `src/experiments/orm/`. Uses upstream's `src/experiments` convention instead of a top-level `experiments/`.
  - **Scripts:** PREREGISTRATION.md (draft with [FILL] fields), `build_dataset`, `judge_baseline`, `train_orm` (GPU only, untested here), `baselines`, `score` (R16 rule), `pilot_gate`.
  - **Tests:** 299 pass.
  - **Judge:** OpenAI gpt-5.5 for now. Adi accepted a deviation from D14 (shared family with the CSAT rater), recorded in PREREGISTRATION Deviations; a non-OpenAI rerun is needed before any external claim.
- Pilot config `src/experiments/orm/configs/pilot.yaml` is ready (53b9e03, hash 83773f5c), not yet run. Cap is $110 (deviation 2 from D15): one gpt-5 agent call cost $0.255.
- **Pilot 1 complete (2026-10-28):** 60 days, 716 calls, $96.19, 0 infra errors. The Mac's lid-close sleep and a restart interrupted it; it resumed cleanly. **Gate FAIL, 3 of 4 checks:**
  - churn 10.2% per branch-month (band ≤10%): FAIL, borderline
  - 7-day repeat contact 41% (band 10-30%): FAIL. Driven by organic contacts at 1/30 days, which alone give about 21% repeat within 7 days.
  - transcript-count logistic AUROC 0.54, CI [0.40, 0.68]: FAIL
  - best-minus-worst variant churn diff 16.7pp, CI [3.3, 30]: PASS (g5-f0 best, g41-f10 worst)
  - **Diagnosis of the signal fail:** the signal exists, but the count features can't see whether the issue was resolved.
    - Task failure gives churn AUROC 0.73: churn is 53% after failed calls vs 14% after solved ones.
    - The oracle direct contribution gives 0.735; support-only calls give 0.70.
  - **Measured:**
    - $0.134 per call
    - p95 transcript is 4,955 tokens (the 8k ORM context is fine)
    - 433 mature rows, 138 positive
    - 54 of 270 branches churned or downgraded in 60 days
  - **Sizing problem:** 200 distinct test events (D17) needs about x25 scale: ~740 customers, ~17.7k calls, about $2.4k.
  - **3 upstream `test_cli_output` failures:** pre-existing on this machine (voice extra not installed).

Phases 5 and 6 can run in parallel with phases 2-4 once phase 1 freezes the ledger schema (see Worktree parallelization).

## Evaluation path (Adi, 2026-10-27)

Goal beyond prediction: use ORMs to improve interactions and steer bad agent trajectories. Two evaluation stages:
1. **Sim 1 (full run): prediction.** Customer-level split (built), ORM vs LLM judge (R16).
2. **Freeze the ORM, then Sim 2: transfer and steering.** New seed and new customers, optionally D16's perturbed hazard. Matched groups: unsteered agents vs the same agents steered by the frozen ORM (best-of-n reply selection or escalation). Measure true churn difference and ORM accuracy on unseen customers.

Needed before Sim 2:
- **Prefix rows in the export:** one row per agent turn with the final outcome label, so the ORM can score partial conversations (~1h).
- **Steering wrapper** as an extra campaign variant (~3-4h).
- **Gaming check:** steering must lower true churn, not only predicted churn; blocked cancels already count as churn.

## Realism backlog (revisit at the end; current assumptions kept for now)

Unrealistic parts that can change the measured ORM-vs-judge result (from the pilot audit, 2026-10-27):
1. **Churn comes from a 5-input formula** (task solved, CSAT, tool errors, repeat contacts, price) that never reads the conversation. A predictor can win by learning those inputs instead of call quality.
2. **CSAT is the only path from conversation quality to churn, and it's weak.** The rater is a lenient LLM that doesn't know if the goal was met (it gave 5/5 to a failed call). It's the same model family as the judge (Deviation 1).
3. **Shortcuts:**
   - Every injected failure shows the identical string "Service temporarily unavailable".
   - Cancel calls are near-free positives because the customer says "cancel".
   - Fixes to consider: vary the error text; score support calls separately from intent calls.
4. **Simulated customers are unrealistically uniform:** polite, cooperative LLM customers with only 2 real personalities, all on the same account (same lines, bills and phone). Real traces are messier and carry history.
5. **The gap between agents is exaggerated** (gpt-4.1 vs gpt-5 is larger than production variation), which inflates AUROC compared with a real deployment.

6. **Retention follows a fixed rule** (accept if price sensitivity > 0.5). Pilot 1 saved 22 cancel calls and cancelled 2. Real save rates are lower and depend on how the agent handles the call.
7. **Contact frequency is too high:** organic issues at 1 per customer per month, so 41% of calls are followed by another within 7 days.
8. **"Resolved" flips meaning on intent calls:** a resolved cancel call means the line closed (churn). Any single resolution signal must be read per call type.

Possible fixes noted so far: make retention depend on agent handling; lower organic contact rate; score support vs intent calls separately; vary injected error messages; give the CSAT rater the customer's goal and whether it was resolved.

## Showcase UI spec (item 6; design review 2026-10-27, all decisions auto-taken as recommended per Adi's D3 answer, reversible)

**Who it's for (1A):** primary = someone being shown the experiment (Adi's team, a prospect, a reviewer) who must grasp in 30 seconds that agent quality changes business outcomes. Secondary = a researcher debugging labels. Search, diagnostics and raw tool payloads sit one click deeper, never on the first screen.

**Surface type (4A):** OPERATE (app UI): table-led, calm and dense, with no cards in the main workspace. Hierarchy comes from typography, whitespace and thin rules. Copy uses utility language (orientation, status, action).

**Routes and navigation (1C):** 4 routes. Each URL holds its full state (campaign, variants, customer, as-of day, query, filters), so a shared link reproduces the view.

```
 top bar (every page): "Lifecycle Lab" · campaign <select> · as-of virtual day · Overview | Compare | Search · [search box]
 ┌──────────────┬──────────────────────────────┬────────────────────────────┬──────────────────────────┐
 │ /  Overview  │ /compare?pop=&a=&b=          │ /c/{pop_id}?v=&asof=       │ /search?q=&v=&day=       │
 │ 3×3 matrix   │ one customer, two variants,  │ one customer branch:       │ hits → exact message or  │
 │ + variant    │ shared day axis, divergence  │ state-as-of + timeline +   │ tool call, context kept  │
 │   table      │ marked                       │ selected conversation      │                          │
 └──────┬───────┴──────────────┬───────────────┴──────────────┬─────────────┴──────────────────────────┘
        └ cell click → variant table   "Compare matched customers" (primary action)   any event → /c/... at that day
```

**Overview (1B), in order:**
1. One line naming what this is: "Synthetic telecom lifecycle experiment · 9 agent variants · same 30 customers". Next to it, campaign status and data cutoff ("virtual day 60 of 60 · complete").
2. **Visual anchor:** a 3×3 matrix with agent model (rows) by tool-failure rate 0 / 10% / 30% (columns). Each cell shows 28-day churn-or-downgrade % with its 95% CI and "n mature / n total". Cell tint is a single-hue scale. The number is always printed; color is never the only cue.
3. Variant table, one row per variant. Columns:
   - task success %
   - mean CSAT (n)
   - 7-day repeat contact %
   - 28-day churn/downgrade % (CI)
   - pending windows
   - infra errors
   - $ cost
4. "Featured comparison": one matched customer under two variants, with the divergence day marked. Its link is the primary action, **Compare matched customers**.

**Customer page (1D):** the context bar always shows campaign · population customer · variant · "as of virtual day N".
- Left column: product state as of that day, as a compact definition table: plan, lines, bill status, tenure, last CSAT. Selecting an event shows before/after differences.
- Main area: the lifecycle timeline. Below it, the selected conversation.

**Timeline encoding (4A):**
- x axis = virtual day; one lane per branch.
- Glyphs:
  - ● task solved
  - ○ task failed
  - red tick = tool error
  - CSAT digit
  - ↺ returned for help
  - ▼ downgrade
  - ✕ churn (a terminal marker; the lane ends there)
- A thin band after each conversation shows its 28-day window. An open window is hatched and labeled "open · matures day 34".
- Every glyph has a text label on hover/focus and an entry in the equivalent event table (6B).

**Interaction → outcome evidence (3A):**
- Default chain: conversation → recorded tool calls and state changes → later lifecycle events → window label.
- Each step shows its virtual day and its ledger event id.
- The wording is "within 28 days of", never "caused".
- Above every transcript, a strip shows what followed: "CSAT 2 → returned for help d+2 → downgrade d+4 · churn window open". Each item links to its spot on the timeline (3B).

**Compare (7C):**
- Default: two agent models at the same failure rate, on the same matched customer. Any 2 of the 9 variants can be chosen; when both factors differ, a "factors differ: model and failure rate" note is shown.
- Shared day axis with synchronized scrolling.
- Divergence = the first virtual day the two event streams differ. The page opens scrolled to it, with a label such as "A: task failed, 2 tool errors · B: solved". A "Next divergence" control moves on.
- A missing observation shows as an explicit gap, never as an empty lane.
- Customer-level evidence is visually separate from cohort numbers.

**Search (1C):**
- FTS5 results show a highlighted excerpt, customer, variant, virtual day, speaker or tool name, and conversation id.
- A result links to the exact message or tool call; returning keeps the query, filters and scroll position.
- "No matches", "empty ledger" and "search failed" are three different states.
- Results paginate at 50.

**Simulator truth layer (7A):**
- Off by default. Turned on only by `tau2 lifecycle serve --oracle`.
- When on, hidden traits, hazard terms and injected failures appear in a separate, hatched "Simulator truth: not visible to ORM or judges" panel, served from separate `/api/oracle/*` endpoints.
- Default API responses carry permitted fields only. The R5 sentinel leakage test also covers the default API.
- Only this layer may show causal attribution ("direct hazard contribution").

**States (2A):**

| Feature | Loading | Empty | Error | Success | Partial / domain |
|---|---|---|---|---|---|
| Overview matrix | grey cell skeletons, labels visible | "No campaign yet. Run `tau2 lifecycle run --config <file>`" with the command copyable | "Can't open ledger at <path>" or "Schema v<a> ≠ app v<b>", with the config hash | numbers + CI + n | pending windows counted and excluded from rates; budget-stopped campaign: banner "stopped at day 41 · $75 cap", KPIs censored after that day, variants with unequal completed days flagged |
| Running campaign | — | — | — | — | banner "day 23 / 60 · running · data as of 14:02" + Refresh button (no auto-poll) |
| Variant table | row skeletons | "No variants in this campaign" | as above | rows | infra-error count per variant; those conversations excluded from task success |
| Timeline | lane skeleton | "No conversations yet for this customer" | inline error with event id | glyphs | open window hatched; churned lane ends at ✕; template-swap note (OV3) on that conversation |
| Conversation | transcript skeleton | — | "Conversation <id> missing from ledger" | turns + tool calls | CSAT pending: "CSAT pending · rater failed · retry queued" (no number); infra error: distinct badge |
| Tool calls | — | — | — | collapsed to one line, monospace | payload > 40 lines: first 12 lines + "Show all (n lines)" |
| Search | inline "Searching…" | "No matches for '<q>'. Try a tool name like refuel_data." | "Search failed: <reason>" | results | >50 results paginated |
| Compare | two lane skeletons | "Pick two variants" with defaults preselected | — | aligned lanes | no divergence: "Identical through day N" |

**Motion (4B):** none, except an instant scroll to the divergence day. If animated at all, ≤200ms ease-out, and none under `prefers-reduced-motion`.

**Design tokens (5A):** new `web/DESIGN.md`. Colors and roles come from Adi's lab-report style. Fonts are IBM Plex Sans (UI) and IBM Plex Mono (IDs, numbers, tool calls, timestamps), self-hosted woff2 for offline use; Inter is avoided as the default UI face.

```css
:root{--paper:#fff;--ink:#111827;--soft-ink:#465166;--quiet:#6b7280;--line:#e2e5e9;
--signal:#00a84f;--fail:#b42318;--pending:#6b7280;--focus:#00a84f;
--sans:"IBM Plex Sans",sans-serif;--mono:"IBM Plex Mono",monospace;
--s1:4px;--s2:8px;--s3:16px;--s4:24px;--s5:40px}
```
- Body 16px, line-height 1.5. Table text 15px with tabular numerals. Metadata never below 13px.
- Radius 4px on inputs only. No shadows, no gradients.
- Variants are identified by direct text labels plus line pattern, never by 9 colors.
- Selection color, caret, scrollbars and focus ring are themed from the tokens.

**Responsive (6A):** desktop-first research tool.
- ≥1280px: full layout.
- 768–1279px: on the customer page, the state table moves above the timeline; compare stays side by side with horizontal scroll.
- <768px: overview shows the matrix and variant table (horizontal scroll); the customer timeline scrolls horizontally with a sticky day axis; compare never stacks variants, it scrolls.

**Accessibility (6B):**
- Landmarks: header, nav, main.
- Tables have a `<caption>` and `th scope`.
- The timeline SVG has `role="img"` with a summary, plus an equivalent keyboard-navigable event table. Arrow keys move between events; Enter opens one.
- Visible 2px focus ring.
- Glyph shape and text are never color-only.
- Contrast ≥4.5:1, checked for --fail and --quiet on white.
- 44px touch targets.
- Search input has a visible label.
- Visited links are distinguishable.

**Fixtures for UI tests (2A, 6B):** one fixture ledger with:
- a mature outcome, an open window and a censored record
- pending CSAT, an infra error and a failed tool call
- a template swap
- divergent branches and a missing observation
- a 200-line tool payload, a churned customer and a budget-stopped campaign

## Scope record

feature answers: D1 item 5 analytics export = Include; D2 item 6 = Include full UI;
D3 item 4 Langfuse tracing = Defer (local per-call role tags agent / simulated-user /
lifecycle kept); base repo (Adi, 2026-10-27 chat, unprompted) = new repo forked from
sierra-research/tau2-bench, no churnbench code or dependency (decision log
6918b02a); structure: B Smaller arrangement (D4).
accepted scope: τ²-bench fork; items 1, 2, 3, 5, 6, 7; item 4 deferred to TODO.
Layout: src/tau2/lifecycle/{world.py, checkpoint.py, kpis.py, campaign.py, export.py,
cli.py}; web/{server.py, index.html, app.js}. campaign.py wraps upstream
src/tau2/runner/batch.py + runner/checkpoint.py; ledger is the single source of truth
and customer state is a projection of it.
pending remedies: R1, R2, R3, R4, R5, R6.

## Decision ledger

### R1: Who decides lifecycle outcomes (label source)
Finding: S1, P0, confidence 8/10, spec item 2 ("let it decide whether to use the
product, return for help, rate the experience, renew, downgrade or cancel"), reviewer
plan-eng-review. If an LLM reads the conversation and decides churn, the ORM learns to
imitate that LLM and the LLM-as-judge baseline is scored on how closely it resembles
the simulator's own model. The comparison then measures model similarity, not outcome
prediction. A same-family judge wins by construction.
Plan baseline: original proposal, LLM lifecycle model decides all outcomes.
Runtime evidence: none (repo not created). Upstream τ²-bench has no lifecycle model.
Comparison grid:
| Choice | Current | A | B | C |
|---|---|---|---|---|
| R1 outcome mechanism | LLM decides all (spec item 2) | authored hazard over objective signals + hidden customer traits + noise; LLM only gives CSAT rating + comment, which feeds the hazard | LLM decides all, with hidden persona traits; decider and judge from different model families | authored model only, CSAT authored too, no LLM after the conversation |
| Base repo / layout / items | approved (scope record) | unchanged | unchanged | unchanged |
| R2-R6 | pending | pending | pending | pending |
Question D5:
D5 — Who decides churn, renewal and repeat contact after a conversation?
Project/branch/task: new τ²-bench fork for the ORM vs LLM-as-judge experiment.
ELI10: The ORM is trained to predict outcomes, so whatever generates those outcomes is the answer key. If an LLM reads the transcript and decides "this customer cancels", the ORM just learns to copy that LLM, and an LLM judge from the same family gets the answer for free. Real customers churn for reasons partly visible in the trace (unresolved issue, wrong refund) and partly not (price sensitivity, a competitor, bad luck). An authored hazard model with hidden traits and noise reproduces that and records exactly why each customer left.
Stakes if we pick wrong: the headline "ORM beats LLM judge" (or the reverse) is explained away as "it matched the simulator's model", and the result cannot be defended.
Recommendation: A) Hybrid because the trace drives part of the outcome, hidden traits drive the rest, and every term is logged as private ground truth.
Completeness: A=9/10, B=5/10, C=8/10
Pros / cons:
A) Hybrid hazard (recommended)
  ✅ Outcome depends on objective trace facts (DB-state task success, errors, repeat contacts) that a judge can learn
  ✅ Hidden traits and noise make some churn unexplainable from the trace, so a perfect score is impossible and meaningful
  ❌ The hazard weights are authored assumptions you must state in any writeup (human: ~3 days / CC: ~3 hrs)
B) LLM decides
  ✅ Closest to the spec wording; rich, varied behaviour without hand-tuned weights
  ✅ Model-family separation reduces, but does not remove, the judge's built-in advantage
  ❌ No per-term ground truth, and the experiment measures LLM-to-LLM agreement (human: ~2 days / CC: ~2 hrs)
C) Authored only
  ✅ Cleanest answer key, fully reproducible, zero lifecycle LLM cost per checkpoint
  ❌ CSAT becomes a formula, which makes the CSAT target trivially learnable and less realistic (human: ~2 days / CC: ~2 hrs)
Net: a defensible answer key with authored weights versus spec-faithful LLM behaviour that undermines the comparison.
Header: Label source
Options:
A) Hybrid hazard (recommended)
Authored per-checkpoint hazard for churn, downgrade, renew and return-for-help over objective signals logged from the run (DB-state task success, unresolved issues, repeat contacts, tool errors, turns) plus hidden customer traits (patience, price sensitivity, baseline churn) plus seeded noise. The LLM lifecycle call only writes a 1-5 CSAT rating and comment from the customer's view; that rating is one hazard input. Every hazard term is logged as private oracle data.
B) LLM decides
LLM lifecycle model decides every action at each checkpoint from the transcript, customer profile and hidden traits; decisions validated then executed. The LLM-as-judge baseline must use a different model family than the lifecycle model.
C) Authored only
No LLM after the conversation. Authored hazard over objective signals and hidden traits decides every action, and CSAT is computed from the same signals.

State: approved
Actual answer: A) Hybrid hazard (recommended), D5 answer 2026-10-27
Accepted scope: Authored per-checkpoint hazard for churn, downgrade, renew and return-for-help over objective signals logged from the run (DB-state task success, unresolved issues, repeat contacts, tool errors, turns) plus hidden customer traits (patience, price sensitivity, baseline churn) plus seeded noise. The LLM lifecycle call only writes a 1-5 CSAT rating and comment from the customer's view; that rating is one hazard input. Every hazard term is logged as private oracle data. Lives in src/tau2/lifecycle/checkpoint.py.
History: none

### R2: The ORM vs LLM-judge experiment itself is not in the plan
Finding: S2, P1, confidence 9/10, purpose statement ("fine-tune an outcome reward model ... compare it to just a regular llm as a judge") vs spec items 1-7, which end at "Export permitted traces and mature KPI labels". No item trains the ORM, runs the judge, or defines how the winner is scored. Without pre-set splits and metrics, the result can be tuned after the fact.
Plan baseline: original proposal, no experiment harness.
Runtime evidence: upstream τ²-bench has src/tau2/evaluator/review_llm_judge.py (an LLM reviewer pattern to reuse for the judge baseline). No ORM code anywhere.
Comparison grid:
| Choice | Current | A | B |
|---|---|---|---|
| R2 experiment harness | absent | in this repo: experiments/orm/ with dataset builder (customer-level splits), LLM-judge baseline, ORM training script, preregistered metrics + baselines | not in this plan; repo stops at the export, harness planned separately later |
| ORM base model + training infra | unspecified | unspecified, pending (asked after R2) | unspecified, pending |
| R1 | approved hybrid hazard | unchanged | unchanged |
| R3-R6 | pending | pending | pending |
Question D6:
D6 — Add the ORM vs LLM-judge experiment harness to this plan?
Project/branch/task: new τ²-bench fork for the ORM vs LLM-as-judge experiment.
ELI10: The 7 items build the data factory but stop at "export the labels". Nothing trains the ORM, runs the LLM judge, or says how the winner is decided. If those rules are written after seeing results, it is easy to pick the metric where your side wins. Writing them down first (which customers are held out, which metrics, which dumb baselines) makes the result trustworthy.
Stakes if we pick wrong: without the harness the repo produces data but no answer; without preregistered splits, a customer seen in training leaks into the test set and the ORM looks better than it is.
Recommendation: A) Include because the comparison is the whole point and the split and metric rules must exist before the first dataset is generated.
Completeness: A=9/10, B=5/10
Pros / cons:
A) Include (recommended)
  ✅ Splits by customer, not by conversation, so no customer appears in both train and test
  ✅ Dumb baselines (task-success only, logistic on objective features, random) show whether either judge adds anything
  ❌ Adds ~5 files and a GPU or fine-tuning-API dependency for the ORM step (human: ~1 week / CC: ~4-6 hrs)
B) Separate later
  ✅ Keeps this plan to the simulator; the experiment gets its own focused plan
  ✅ Lets you look at the first dataset before committing to an ORM base model
  ❌ Data-format decisions get made without the consumer, so the export may need rework
Net: the answer and its scoring rules now versus a smaller simulator-only plan.
Header: Experiment
Options:
A) Include (recommended)
Add experiments/orm/: build_dataset.py (customer-level train/val/test split, fixed seed, written before any labels are generated), judge_baseline.py (LLM-as-judge with an outcome rubric, reusing the evaluator/review_llm_judge.py pattern), train_orm.py (fine-tune a small model on permitted traces → mature labels), score.py (AUROC + Brier/calibration for churn and renewal, Spearman for CSAT, top-k capture of truly harmful traces), PREREGISTRATION.md. Baselines: task-success-only, logistic regression on objective features, random. ORM base model and training infra stay pending as a separate choice.
B) Separate later
This plan stops at the training export (item 7). The experiment harness gets its own plan after the first dataset exists. Logged as a TODO.

State: approved
Actual answer: A) Include (recommended), D6 answer 2026-10-27
Accepted scope: Add experiments/orm/: build_dataset.py (customer-level train/val/test split, fixed seed, written before any labels are generated), judge_baseline.py (LLM-as-judge with an outcome rubric, reusing the evaluator/review_llm_judge.py pattern), train_orm.py (fine-tune a small model on permitted traces → mature labels), score.py (AUROC + Brier/calibration for churn and renewal, Spearman for CSAT, top-k capture of truly harmful traces), PREREGISTRATION.md. Baselines: task-success-only, logistic regression on objective features, random. ORM base model and training infra stay pending as a separate choice (R7).
History: none

### R3: Which trace gets the blame (per-trace label)
Finding: S3, P1, confidence 8/10, spec item 7 ("Export permitted traces and mature KPI labels") and purpose ("how agent traces will affect business outcomes"). A customer has several conversations before churning. The plan never says which trace a later churn is attributed to, so the ORM's training target is undefined.
Plan baseline: original proposal, label assignment unspecified.
Runtime evidence: none (repo not created). R1 (approved) logs every hazard term per checkpoint as private oracle data, which makes exact attribution available at no extra run cost.
Comparison grid:
| Choice | Current | A | B | C |
|---|---|---|---|---|
| R3 training label | unspecified | outcome window after each trace (churn ≤28 virtual days, repeat contact ≤7 days, next CSAT) | same window labels | same window labels |
| R3 evaluation target | unspecified | window labels + private oracle attribution (the trace's own hazard-term contribution from R1), evaluation only, never in training rows | window labels only | window labels + counterfactual pairs: re-run held-out target conversations with a second agent variant on the same customer and seeds; outcome difference = causal label |
| Run cost | n/a | no extra | no extra | ~2x conversations on the test set |
| Censoring | spec item 3 (approved) | traces whose window has not closed are marked immature and excluded | same | same |
| R1, R2 | approved | unchanged | unchanged | unchanged |
| R4-R7 | pending | pending | pending | pending |
Question D7:
D7 — How does each trace get its outcome label?
Project/branch/task: new τ²-bench fork for the ORM vs LLM-as-judge experiment.
ELI10: A customer might talk to the agent three times and then cancel. Which conversation caused it? For training, use what a real company would have: what happened in the weeks after each conversation. For scoring, the simulator knows the true answer, because the hazard model (D5) logs how much each conversation pushed churn up. Using that secret answer only for grading tests whether a judge spots the truly harmful trace, not just a trace that happened before a cancellation.
Stakes if we pick wrong: with window labels alone, a harmless first chat gets blamed for a later bad one, and both judges are graded on noisy blame.
Recommendation: A) Window + oracle because exact attribution is already logged by D5 at zero extra cost, and keeping it out of training rows preserves the realistic setup.
Completeness: A=9/10, B=6/10, C=10/10
Pros / cons:
A) Window + oracle (recommended)
  ✅ Trains on labels a real company could collect, so the method transfers in principle
  ✅ Grades on exact simulator ground truth with no extra conversations
  ❌ Oracle attribution reflects the authored hazard weights, which must be stated in any writeup (human: ~1 day / CC: ~1 hr)
B) Window only
  ✅ Simplest; one label set for training and grading
  ✅ Mirrors exactly what production data looks like
  ❌ Grading inherits blame noise, so a better judge can look no better
C) Window + counterfactual
  ✅ Causal labels independent of the hazard formula: same customer, same seeds, different agent
  ✅ Strongest claim for a writeup
  ❌ About twice the conversation spend on the test set, plus pairing logic (human: ~3 days / CC: ~3 hrs)
Net: exact free ground truth for grading versus simpler or more expensive alternatives.
Header: Trace label
Options:
A) Window + oracle (recommended)
Training rows carry window labels only: churn or downgrade within 28 virtual days, repeat contact within 7 days, next CSAT. Traces with an open window are marked immature and excluded. Evaluation additionally scores against private oracle attribution (each trace's logged hazard-term contribution from D5), which never appears in training exports.
B) Window only
Window labels (same definitions) for both training and evaluation. No oracle attribution used.
C) Window + counterfactual
Window labels for training. For the test set, re-run each target conversation with a second agent variant on the same customer and seeds; the outcome difference is the causal label. About 2x conversation cost on the test set.

State: approved
Actual answer: A) Window + oracle (recommended), D7 answer 2026-10-27
Accepted scope: Training rows carry window labels only: churn or downgrade within 28 virtual days, repeat contact within 7 days, next CSAT. Traces with an open window are marked immature and excluded. Evaluation additionally scores against private oracle attribution (each trace's logged hazard-term contribution from D5), which never appears in training exports.
History: none

### R4: Traces need real quality spread, or there is nothing to learn
Finding: S4, P1, confidence 7/10, spec item 7 ("matched customer populations across agent variants") names variants but not what makes them differ. With one strong agent most conversations succeed, labels collapse to "no churn", and outcomes are driven by hidden traits. Both judges then look equally useless.
Plan baseline: original proposal, agent variants unspecified.
Runtime evidence: upstream τ²-bench runs one agent LLM per run (src/tau2/agent/llm_agent.py). Published τ²-bench pass rates differ widely by model, so model choice alone gives some spread.
Comparison grid:
| Choice | Current | A | B |
|---|---|---|---|
| R4 agent variants | unspecified | 3 agent models of different strength × 3 injected tool-failure rates (0, 0.1, 0.3) = 9 variants, matched customer populations | 3 agent models only, no injection |
| R1-R3 | approved | unchanged | unchanged |
| R6, R7 | pending | pending | pending |
Question D8:
D8 — What makes agent variants differ in quality?
Project/branch/task: new τ²-bench fork for the ORM vs LLM-as-judge experiment.
ELI10: The ORM can only learn "this trace hurts the business" if some traces actually do. Using three agent models of different strength gives natural variety. Also randomly failing a share of tool calls (0%, 10%, 30%) adds controlled, known failures, so you know exactly which traces were damaged and by how much.
Stakes if we pick wrong: too little spread means a dataset of mostly happy customers, and neither judge can be told apart from the random baseline.
Recommendation: A) Models + injection because injected failures give a known, tunable dose of bad traces on top of natural model differences.
Completeness: A=9/10, B=7/10
Pros / cons:
A) Models + injection (recommended)
  ✅ Failure rate is a dial: if labels are too skewed, turn it up without changing models
  ✅ Injected failures are known ground truth, which helps debug the hazard and the labels
  ❌ Injected tool errors are less natural than real model mistakes; judges might learn the error string (human: ~2 days / CC: ~2 hrs)
B) Models only
  ✅ Every failure is a real model mistake, the most realistic trace texture
  ✅ Fewer variants means fewer runs for the same per-variant sample size
  ❌ No dial: spread depends on which models you pick and how they behave on the chosen domain
Net: a controllable failure dial versus purely natural mistakes.
Header: Variants
Options:
A) Models + injection (recommended)
Agent variants = 3 agent models of different strength (exact models chosen at Phase 0 from current τ²-bench scores on the chosen domain) × injected tool-call failure rate {0, 0.1, 0.3}. Injection is a seeded wrapper on the environment's tool execution that returns a realistic error; injected calls are logged as private oracle data and stripped of any marker in training exports. Same customer population for all 9 variants.
B) Models only
Agent variants = 3 agent models of different strength, no injection. Same customer population for all 3 variants.

State: approved
Actual answer: A) Models + injection (recommended), D8 answer 2026-10-27
Accepted scope: Agent variants = 3 agent models of different strength (exact models chosen at Phase 0 from current τ²-bench scores on the chosen domain) × injected tool-call failure rate {0, 0.1, 0.3}. Injection is a seeded wrapper on the environment's tool execution that returns a realistic error; injected calls are logged as private oracle data and stripped of any marker in training exports. Same customer population for all 9 variants.
History: none

### R6: Domain and upstream version
Finding: S6, P2, confidence 8/10, spec item 1 ("subscriptions, usage and support history") plus "Starting from a fresh τ-bench clone". Upstream τ²-bench v2.0.0 (released 2026-10-20) ships airline, retail, telecom, banking_knowledge, intake, mock and voice code paths. The plan names neither a domain nor a version. Each domain needs its own hazard signals and customer state, and an unpinned clone drifts under the runner hook (D4 couples campaign.py to runner internals).
Plan baseline: original proposal, domain and version unspecified.
Runtime evidence: upstream tree read via GitHub API 2026-10-27: src/tau2/domains/{airline,retail,telecom,banking_knowledge,intake,mock}; latest release v2.0.0 2026-10-20. Telecom's data model has customers, lines, plans and bills; retail has orders and returns; airline has reservations and membership tiers.
Comparison grid:
| Choice | Current | A | B | C |
|---|---|---|---|---|
| R6 domain | unspecified | telecom only | retail only | telecom + retail |
| R6 upstream pin | unpinned | tag v2.0.0 | tag v2.0.0 | tag v2.0.0 |
| R1-R4 | approved | unchanged | unchanged | unchanged |
| R7 | pending | pending | pending | pending |
Question D9:
D9 — Which τ²-bench domain does the lifecycle sim run on?
Project/branch/task: new τ²-bench fork for the ORM vs LLM-as-judge experiment.
ELI10: τ²-bench has several "shops" the agent works in. Telecom already has customers with phone plans, lines and bills, so renew, downgrade and cancel are real database changes the agent's tools can touch. Retail has orders and returns but no subscription, so you would have to invent one. Starting with one domain means one set of churn signals to tune. Pinning to release v2.0.0 stops upstream changes from breaking the fork.
Stakes if we pick wrong: a domain without subscriptions makes the business outcomes feel bolted on; two domains doubles the hazard tuning before the first result.
Recommendation: A) Telecom because subscriptions, plans and billing already exist as real state with tools.
Completeness: A=8/10, B=7/10, C=10/10
Pros / cons:
A) Telecom (recommended)
  ✅ Plans, lines and bills already in the DB, so lifecycle actions are real state changes
  ✅ Troubleshooting tasks produce repeat contacts naturally when the fix does not stick
  ❌ Telecom is dual-control (the user also runs device tools), so conversations are longer and cost more (human: ~1 week / CC: ~4 hrs for state + hazard)
B) Retail
  ✅ The domain most τ-bench papers cite, including the simulated-user fidelity paper (arXiv 2601.17087)
  ✅ Shorter conversations, cheaper per trace
  ❌ No subscription; renew/downgrade/cancel need an invented membership layer
C) Telecom + retail
  ✅ Tests whether the ORM generalizes across domains, a stronger result
  ❌ Two customer models and two hazard calibrations before any result (human: ~2 weeks / CC: ~8 hrs)
Net: natural subscription state versus citation familiarity or cross-domain breadth.
Header: Domain
Options:
A) Telecom (recommended)
Telecom only. Fork pinned to upstream tag v2.0.0. Customer state extends telecom's existing customer/line/plan/bill records. Retail as a second domain logged as a TODO.
B) Retail
Retail only. Fork pinned to upstream tag v2.0.0. Adds an authored membership subscription to retail customers so renew/downgrade/cancel have state.
C) Telecom + retail
Both domains. Fork pinned to upstream tag v2.0.0. Separate customer-state extensions and hazard calibrations; ORM evaluated in-domain and cross-domain.

State: approved
Actual answer: A) Telecom (recommended), D9 answer 2026-10-27
Accepted scope: Telecom only. Fork pinned to upstream tag v2.0.0. Customer state extends telecom's existing customer/line/plan/bill records. Retail as a second domain logged as a TODO.
History: none

### R7: ORM base model and training setup
Finding: S7, P2, confidence 7/10, R2 accepted scope ("ORM base model and training infra stay pending as a separate choice"). Telecom conversations are long (dual-control, device tool calls), so input length is a real constraint, and "outcome reward model" usually means an LM backbone with a scalar head, reusable later as an RL reward.
Plan baseline: R2 approved train_orm.py with model and infra unspecified.
Runtime evidence: none (no local GPU checked; Adi's compute unknown). Telecom transcript lengths unmeasured; Phase 0 measures them.
Comparison grid:
| Choice | Current | A | B | C |
|---|---|---|---|---|
| R7 ORM backbone | unspecified | small open-weight decoder (~1-4B), LoRA, multi-head output (P(churn 28d), P(repeat contact 7d), CSAT) | long-context encoder (~400M, 8k tokens) with classification/regression heads, full fine-tune | hosted fine-tuning API on a small closed model, labels as generated tokens |
| R7 training infra | unspecified | rented single GPU (on-demand), HF transformers + peft | same rented GPU or a large local Mac, HF transformers | provider API, no GPU |
| R1-R4, R6 | approved | unchanged | unchanged | unchanged |
Question D10:
D10 — What is the small ORM, and where does it train?
Project/branch/task: new τ²-bench telecom fork for the ORM vs LLM-as-judge experiment.
ELI10: The ORM is the small model you fine-tune to read a trace and output "chance this customer churns, chance they call back, expected rating". A small open model with a numeric head (A) is what "reward model" normally means, and you can reuse it later to train agents. An encoder (B) is cheaper and faster but reads fewer tokens, so long telecom chats may be cut off. A hosted API (C) needs no GPU but gives you text answers, not clean probabilities, and you cannot reuse it as a reward.
Stakes if we pick wrong: a model that truncates traces loses the part where the failure happens; a closed API model blurs the "small judge vs big LLM judge" comparison.
Recommendation: A) Open decoder because it handles long traces, outputs calibrated probabilities, and doubles as an RL reward later.
Note: options differ in kind, not coverage — no completeness score.
Pros / cons:
A) Open decoder + LoRA (recommended)
  ✅ Long context covers full telecom traces; numeric heads give probabilities you can calibrate
  ✅ Open weights: reusable as an RL reward and runnable inside Proper Motion later
  ❌ Needs a rented GPU and some training plumbing (human: ~1 week / CC: ~4 hrs; GPU roughly tens of dollars per run)
B) Encoder classifier
  ✅ Fastest and cheapest to train and serve; very stable fine-tuning
  ✅ Can run on a single small GPU or a large Mac
  ❌ ~8k-token limit risks cutting long telecom traces, and it is not a generative reward model
C) Hosted fine-tune API
  ✅ No GPU or training code; fastest path to a first number
  ❌ Text outputs, weak calibration, closed weights; "small judge" becomes a smaller cousin of the LLM judge
Net: a reusable, calibrated reward model on a rented GPU versus cheaper or GPU-free shortcuts.
Header: ORM model
Options:
A) Open decoder + LoRA (recommended)
Small open-weight decoder (~1-4B; exact model picked at Phase 4 from current options with context ≥ the measured 95th-percentile trace length), LoRA fine-tune with three output heads: P(churn or downgrade ≤28d), P(repeat contact ≤7d), CSAT. Trained on a rented single GPU with HF transformers + peft; training config and seed versioned with the dataset.
B) Encoder classifier
Long-context encoder (~400M, ~8k tokens) full fine-tune with the same three heads; traces longer than the window are truncated from the start, keeping the end of the conversation. Rented small GPU or local Mac.
C) Hosted fine-tune API
Fine-tune a small hosted model through a provider API to emit the three labels as text; probabilities taken from token logprobs where available.

State: approved
Actual answer: A) Open decoder + LoRA (recommended), D10 answer 2026-10-27
Accepted scope: Small open-weight decoder (~1-4B; exact model picked at Phase 4 from current options with context ≥ the measured 95th-percentile trace length), LoRA fine-tune with three output heads: P(churn or downgrade ≤28d), P(repeat contact ≤7d), CSAT. Trained on a rented single GPU with HF transformers + peft; training config and seed versioned with the dataset.
History: none

Scope Challenge result: scope reduced per recommendation (item 4 Langfuse deferred); experiment harness added (D6). R1-R7 resolved.

### R5: Training exports leak private simulator information (no question: approved contract)
Finding: S5, P1, confidence 8/10, spec item 7 ("Export permitted traces and mature KPI labels separately from private simulator information"). This is already an approved requirement; the finding names what "private" must cover in τ²-bench so implementation cannot miss it. Upstream task objects carry the user's hidden scenario/instructions and evaluation criteria (expected actions, DB-state checks), and simulation results carry reward_info. Any of these in a judge's input hands it the answer.
Disposition: carried forward as necessary implementation of item 7. Private (never in training or judge input): hidden customer traits, hazard terms and oracle attribution (D5, D7), injection markers (D8 if approved), task user_scenario/instructions, evaluation_criteria, reward_info, simulated-user and lifecycle-model prompts. Permitted: agent-visible messages and tool calls/results, agent system prompt, timestamps, customer and conversation IDs. Required proof: an allowlist export function plus a leakage test that seeds sentinel strings into every private field and asserts none reach the export.
State: approved (spec item 7)

## Review Section 1: Architecture

Evidence read 2026-10-27 from upstream sierra-research/tau2-bench main (v2.0.0):
- `src/tau2/domains/telecom/environment.py:97` `get_environment(db: Optional[TelecomDB] = None, user_db=..., ...)` accepts an injected DB (deep-copied).
- `src/tau2/evaluator/evaluator_env.py:89-123` builds predicted and gold environments with the same `initialization_data` + `initialization_actions`, replays `evaluation_criteria.actions` on gold, compares DB hashes.
- `data/tau2/domains/telecom/db.toml`: 4 customers. `tasks.json`: 2,285 tasks; every task's initial state targets customer `C1001`; 2,256 tasks have gold actions carrying `customer_id`.
- `src/tau2/runner/batch.py:19` ThreadPoolExecutor; `:532` "workers only return results" to a single checkpoint writer.

Planned flow (approved scope, smaller arrangement):

```
 campaign.py (config hash, budget, cancel/resume, 9 variants x same population)
     |  for each virtual day d:
     v
 world.py  --- ledger (SQLite WAL, single writer = campaign) ---> state projection per customer
     |  customers due for contact on day d (hazard: return-for-help)
     v
 task instantiation: telecom template -> retarget IDs to customer -> prepend ledger-derived
     |               state as initialization_data/actions           [A2]
     v
 upstream runner/batch.py (threads)  -> Simulation (agent | sim-user | env + tool-failure
     |                                   injection wrapper, D8)
     v  results returned to the single writer
 checkpoint.py: native reward (DB-hash grade) + LLM CSAT rating -> authored hazard (D5)
     |          -> validated actions: use / return / renew / downgrade / cancel
     v
 ledger events (public)  +  oracle rows (private: traits, hazard terms, injections)
     |                                   |
     v                                   v
 kpis.py (CSAT, repeat contact,      export.py: training rows (allowlist, mature windows,
 churn, retention, usage, censoring)            D7 labels)  |  Mixpanel/Amplitude sink (item 5)
     |                                   |
     v                                   v
 web/server.py (read-only, 127.0.0.1)   experiments/orm/ (split, judge, ORM, score)
```

Findings:
1. [P1] (confidence: 9/10) data/tau2/domains/telecom/tasks.json — every one of 2,285 tasks targets customer C1001 and the DB holds 4 customers, so "same customer and account state across conversations" (item 1) has no native support. Quote: initial_state `{"env_type": "assistant", "func_name": "enable_roaming", "arguments": {"customer_id": "C1001", "line_id": "L1002"}}`. → R8 (D11).
2. [P1] (confidence: 8/10) src/tau2/runner/batch.py:19 — upstream batch treats tasks as independent, but one customer's conversations must run in virtual-time order because each depends on the previous outcome. Parallelism must be across customers only. → R9.
3. [P2] (confidence: 8/10) batch.py:532 single checkpoint writer — the ledger must have exactly one writer (campaign thread), SQLite WAL. Necessary implementation of approved items 3 and 7; no question.
4. [P2] (confidence: 7/10) checkpoint.py (proposed) — if the LLM CSAT call fails, a defaulted rating would silently corrupt labels. Required: retry with backoff, then mark the checkpoint `pending` and resume it later; never write a default CSAT. Necessary implementation of item 3 ("pending outcomes"); no question.
5. [P2] (confidence: 7/10) web/server.py + export.py (proposed) — UI binds 127.0.0.1 and opens the ledger read-only; the Mixpanel/Amplitude sink goes through the same allowlist as training exports (R5) and reads keys from env vars only. Necessary implementation of items 5, 6, 7; no question.

### R8: Persistent customers on telecom tasks that all target C1001
Finding: Section 1 #1, P1, confidence 9/10, data/tau2/domains/telecom/tasks.json, reviewer plan-eng-review.
Plan baseline: item 1 approved ("Preserve the same customer and account state across conversations"); mechanism unspecified.
Runtime evidence: see Section 1 evidence. get_environment accepts an injected DB; evaluator replays the same init for predicted and gold envs, so init-time changes keep native grading valid.
Comparison grid:
| Choice | Current | A | B |
|---|---|---|---|
| R8 persistence mechanism | unspecified | synthetic customers injected via initialization_data; each task template retargeted (customer/line IDs, name, phone in init actions, gold actions, user scenario); customer's ledger-derived state prepended as init actions; preflight gold-replay check per (customer, task) | overlay: tasks run unchanged on C1001; lifecycle customer is a separate record the agent never sees |
| Native task grading | upstream DB-hash replay | unchanged, valid because both envs replay the same init | unchanged |
| R9 | pending | pending | pending |
Question D11:
D11 — How do different customers persist when every telecom task is written for one customer?
Project/branch/task: new τ²-bench telecom fork (v2.0.0) for the ORM experiment.
ELI10: All 2,285 telecom tasks are about the same customer, C1001. To have many customers with their own history, each task has to be rewritten on the fly for the right customer: swap the IDs and name, and replay that customer's past changes (a plan downgrade last month, an unpaid bill) before the chat starts. τ²-bench grades by replaying the correct actions from the same starting point, so this keeps its grading valid. The alternative keeps tasks untouched and tracks the customer only on the side, but then every conversation is about "John Smith" and the agent never sees the customer's real account.
Stakes if we pick wrong: with the overlay, traces carry no customer-specific signal, so item 1 becomes cosmetic and the ORM cannot learn from account history.
Recommendation: A) Retarget + replay because it gives real per-customer state while keeping τ²-bench's own grading valid.
Completeness: A=9/10, B=4/10
Pros / cons:
A) Retarget + replay (recommended)
  ✅ Agent sees the customer's actual plan, lines, bills and past changes in every conversation
  ✅ Native task success stays comparable to published τ²-bench numbers because grading is untouched
  ❌ ID substitution must cover init actions, gold actions and user instructions; a miss breaks grading (human: ~4 days / CC: ~3 hrs, plus preflight check)
B) Overlay only
  ✅ Zero changes to tasks; native grading trivially intact
  ✅ Fastest path to first conversations
  ❌ Every trace is the same customer; persistence and support history never reach the agent or the trace
Net: real per-customer traces with a substitution step to get right versus a cosmetic persistence layer.
Header: Customers
Options:
A) Retarget + replay (recommended)
Generate N synthetic telecom customers (profile, plan, lines, device defaults) and inject each via initialization_data. Retarget each task template to the customer (customer_id, line_id, name, phone in initialization_actions, gold actions and user_scenario). Prepend the customer's ledger-derived state changes as initialization actions. Before a campaign, a preflight replays gold actions for every scheduled (customer, task) pair; pairs that error are skipped and logged, never run.
B) Overlay only
Run telecom tasks unchanged on C1001. Keep customer profile, subscription and history only in the lifecycle ledger; the agent never sees them.

State: approved
Actual answer: A) Retarget + replay (recommended), D11 answer 2026-10-27
Accepted scope: Generate N synthetic telecom customers (profile, plan, lines, device defaults) and inject each via initialization_data. Retarget each task template to the customer (customer_id, line_id, name, phone in initialization_actions, gold actions and user_scenario). Prepend the customer's ledger-derived state changes as initialization actions. Before a campaign, a preflight replays gold actions for every scheduled (customer, task) pair; pairs that error are skipped and logged, never run.
History: none

### R9: Ordering one customer's conversations in virtual time
Finding: Section 1 #2, P1, confidence 8/10, src/tau2/runner/batch.py:19 (ThreadPoolExecutor over independent tasks), reviewer plan-eng-review.
Plan baseline: D4 approved campaign.py wrapping upstream runner/batch.py + runner/checkpoint.py; scheduling unspecified.
Runtime evidence: upstream batch runs independent tasks concurrently with a single checkpoint writer (batch.py:532). It has no notion of one task depending on another's outcome.
Comparison grid:
| Choice | Current | A | B |
|---|---|---|---|
| R9 scheduling | unspecified | day waves: advance the virtual clock one day; all conversations due that day (all customers, all 9 variants) go to one upstream batch; write results; run checkpoints; advance | per-customer workers: each worker runs one customer's timeline start to end; many in parallel; results queued to the single writer |
| Resume unit | n/a | last completed virtual day | per customer timeline |
| Uses upstream batch.py | D4 approved | yes, unchanged | no, custom worker pool beside it |
| R1-R8 | approved | unchanged | unchanged |
Question D12:
D12 — How are one customer's conversations kept in order while still running in parallel?
Project/branch/task: new τ²-bench telecom fork (v2.0.0) for the ORM experiment.
ELI10: A customer's second call depends on how the first went, so their calls must run in order. Different customers are independent, so they can run at the same time. Day waves move the whole simulated world forward one day at a time and run every call due that day in parallel using τ²-bench's own batch runner. Per-customer workers each walk one customer through their whole timeline, which is faster but needs its own worker pool beside τ²-bench's.
Stakes if we pick wrong: per-customer workers duplicate the runner D4 chose to reuse; day waves wait for the slowest call each day, so long runs take longer.
Recommendation: A) Day waves because it reuses upstream batch and resume unchanged and makes budget and cancel checks one simple step per day.
Note: options differ in kind, not coverage — no completeness score.
Pros / cons:
A) Day waves (recommended)
  ✅ Upstream batch.py and checkpoint.py used as-is, keeping D4's smaller layout honest
  ✅ Resume, budget and cancel all happen at one clean boundary: end of a virtual day
  ❌ Each wave waits for its slowest conversation, so wall-clock time grows with the long tail (human: ~2 days / CC: ~1.5 hrs)
B) Per-customer workers
  ✅ Better throughput: no waiting on other customers' slow calls
  ✅ A crashed customer timeline resumes alone without touching others
  ❌ A second worker pool, budget counter and resume system beside upstream's, which D4 set out to avoid (human: ~4 days / CC: ~3 hrs)
Net: simple, upstream-native scheduling versus faster wall-clock time with more custom machinery.
Header: Scheduling
Options:
A) Day waves (recommended)
campaign.py advances virtual time one day per wave. All conversations due that day, across all customers and all 9 variants, are submitted as one upstream batch (runner/batch.py, unchanged). After the batch returns, the single writer records results, runs lifecycle checkpoints for that day, checks budget and cancel flags, then advances. Resume restarts from the last completed day.
B) Per-customer workers
campaign.py runs a pool of workers, each driving one customer's timeline start to end. Results go to a queue drained by the single ledger writer. A shared spend counter enforces the budget; resume is per customer.

State: approved
Actual answer: A) Day waves (recommended), D12 answer 2026-10-27
Accepted scope: campaign.py advances virtual time one day per wave. All conversations due that day, across all customers and all 9 variants, are submitted as one upstream batch (runner/batch.py, unchanged). After the batch returns, the single writer records results, runs lifecycle checkpoints for that day, checks budget and cancel flags, then advances. Resume restarts from the last completed day.
History: none

Section 1 dispositions: #1 accepted (D11 Retarget + replay); #2 accepted (D12 Day waves); #3, #4, #5 accepted as necessary implementation of approved items 3, 5, 6, 7.

## Review Section 2: Code quality

1. [P1] (confidence: 8/10) lifecycle retargeting (D11) — "C1001" also appears inside free-text user_scenario instructions, so naive string replace can miss or over-replace. Required proof of D11: substitution from an explicit ID map over the parsed task, then assert the serialized retargeted task contains none of the template IDs (C1001, L100x, template name/phone). Necessary implementation; no question.
2. [P1] (confidence: 8/10) kpis.py vs experiments/orm/build_dataset.py (both proposed) — two window-label definitions would drift and make train and eval labels disagree. Required: one `window_labels()` in kpis.py; export.py and build_dataset.py only read its output. Necessary for D7 consistency; no question.
3. [P2] (confidence: 7/10) checkpoint.py (proposed) — hazard weights, trait distributions, window lengths and injection rates belong in the versioned campaign config whose hash stamps every ledger row (item 7), not in code constants. Necessary implementation of item 7; no question.
4. [P2] (confidence: 7/10) campaign.py (proposed) — a cancel or crash mid-wave must not re-pay finished conversations. Required: resume day d through upstream `try_resume` (runner/checkpoint.py) so completed simulations in the interrupted wave are reused. Necessary implementation of D12; no question.
5. [P2] (confidence: 7/10) matched populations (item 7) — random draws must be keyed by (customer, virtual day, purpose), not by call order, so the same customer gets the same task, contact-day and noise draws in all 9 variants until outcomes diverge. Necessary implementation of item 7; no question.

Section 2 dispositions: #1-#5 accepted as necessary implementation of approved D7, D11, D12 and item 7.


## Review Section 3: Tests

Framework: upstream τ²-bench uses pytest (`tests/`); no CLAUDE.md testing section exists for the new repo. All paths below are proposed (repo not created), so every row is a GAP by definition; the diagram states what proof each needs.

```
CODE PATHS                                                   USER FLOWS
[+] lifecycle/world.py                                       [+] Run a campaign (CLI)
  ├── append_event() single writer, idempotent event id        ├── [GAP] [→E2E] offline 3-customer x 2-variant x 14-day campaign
  │     [GAP] duplicate id ignored; second writer refused      │        completes; ledger, KPIs, export produced
  └── project_state(customer, day)                             ├── [GAP] [→E2E] cancel mid-wave, resume: no conversation re-paid
        [GAP] fixture ledger -> expected plan/lines/bills      ├── [GAP] budget hit: stops at day boundary with clear message
[+] lifecycle retarget (D11)                                   └── [GAP] config edited between runs: resume refused, says why
  ├── [GAP] ID map applied to init, gold, user_scenario        [+] Showcase UI (web/)
  ├── [GAP] no template ID left in serialized task             ├── [GAP] [→E2E] cohort view -> customer -> timeline -> transcript
  └── [GAP] [→E2E] preflight: gold replay on tasks_small        ├── [GAP] search returns matching chat/tool logs; empty state shown
        retargeted, reward 1.0; erroring pair skipped+logged   ├── [GAP] side-by-side: same customer, two variants, aligned days
[+] lifecycle/checkpoint.py                                    └── [GAP] ledger missing/empty: clear message, no 500
  ├── [GAP] hazard deterministic for (customer, day, seed)     [+] Analytics export (item 5)
  ├── [GAP] invalid action rejected (renew after cancel,       ├── [GAP] payload: stable insert_id, virtual ts, distinct_id
  │         downgrade below lowest plan)                       └── [GAP] provider 4xx/5xx: retried, then reported; ledger untouched
  ├── [GAP] CSAT LLM failure -> pending, never defaulted       [+] ORM experiment
  └── [GAP] oracle rows only in private table                  ├── [GAP] split: no customer in two splits
[+] tool-failure injection (D8)                                ├── [GAP] score.py metrics on toy inputs with known answers
  └── [GAP] seeded rate within tolerance; logged private       ├── [GAP] [→EVAL] judge prompt: rubric output parses, stable
[+] lifecycle/kpis.py                                          └── [GAP] [→EVAL] CSAT rater prompt: ratings track resolution
  └── [GAP] table-driven: churn, retention, repeat, CSAT,
        usage, immature windows excluded (D7)
[+] lifecycle/export.py
  ├── [GAP] CRITICAL leakage: sentinel in every private field
  │         never reaches training rows or analytics payloads
  └── [GAP] immature traces excluded
[+] campaign.py (D12)
  ├── [GAP] customer's day-N conversation sees day-M<N outcome
  └── [GAP] matched populations: same draws in all 9 variants
[+] upstream regression -> R10

COVERAGE: 0/33 paths tested (repo not created) | GAPS: 33 (6 E2E, 2 eval)
```

Value cards for the critical paths are in the Test Plan artifact. Tests made obsolete by this plan: none (no existing tests in the new code).

### R10: Upstream τ²-bench regression contract
Finding: Section 3 regression rule, P1, confidence 8/10. D11 retargets tasks and D8 wraps tool execution; both sit on the path that computes native task reward, which the plan keeps as a separate measure (item 3) and which must stay comparable to published τ²-bench numbers.
Plan baseline: item 3 approved ("Keep native task success ... as separate measures"); regression contract unspecified.
Runtime evidence: upstream has a pytest suite under tests/ (not run; repo not created).
Comparison grid:
| Choice | Current | A | B |
|---|---|---|---|
| R10 regression contract | unspecified | upstream test suite runs unmodified and green; plus golden equivalence: for every tasks_small task, the retargeted version with empty history and 0 injection gets the same gold-replay reward as the original | upstream test suite runs unmodified and green only |
| Intentional upstream changes | n/a | none except CLI subcommand registration | same |
| R11 | pending | pending | pending |
Question D13:
D13 — What must stay identical to upstream τ²-bench?
Project/branch/task: new τ²-bench telecom fork (v2.0.0) for the ORM experiment.
ELI10: The fork rewrites tasks for each customer and can break tool calls on purpose. If either quietly changes how τ²-bench scores a conversation, your "native task success" numbers stop matching everyone else's. Upstream's own tests catch changes to the core. A second check runs every small telecom task twice, original and rewritten for a fresh customer, and requires the same score from both.
Stakes if we pick wrong: a silent grading change makes every task-success number, and the task-success baseline, wrong without any error.
Recommendation: A) Suite + golden because the equivalence check is the only test that catches a retargeting bug in grading.
Completeness: A=10/10, B=6/10
Pros / cons:
A) Suite + golden (recommended)
  ✅ Catches any retargeting slip that changes scoring, on real tasks, before any paid run
  ✅ Runs offline with gold replay only, no model calls, so it is free and fast
  ❌ About 20 extra task replays per CI run and one more fixture to maintain (human: ~1 day / CC: ~30 min)
B) Suite only
  ✅ Zero extra work; upstream's suite already guards the core
  ❌ Upstream tests never see retargeted tasks, so a grading bug from D11 goes unnoticed
Net: one cheap offline equivalence test versus trusting that retargeting never touches grading.
Header: Regression
Options:
A) Suite + golden (recommended)
Upstream pytest suite runs unmodified and must stay green. Add tests/lifecycle/test_retarget_equivalence.py: for every tasks_small telecom task, the retargeted version (fresh customer, empty history, injection off) must get the same gold-replay reward as the original. The only intended upstream change is registering the lifecycle CLI subcommand.
B) Suite only
Upstream pytest suite runs unmodified and must stay green. No retargeting equivalence test. The only intended upstream change is registering the lifecycle CLI subcommand.

State: approved
Actual answer: A) Suite + golden (recommended), D13 answer 2026-10-27
Accepted scope: Upstream pytest suite runs unmodified and must stay green. Add tests/lifecycle/test_retarget_equivalence.py: for every tasks_small telecom task, the retargeted version (fresh customer, empty history, injection off) must get the same gold-replay reward as the original. The only intended upstream change is registering the lifecycle CLI subcommand.
History: none

### R11: How strong is the LLM-as-judge baseline?
Finding: Section 3 LLM/eval scope, P1, confidence 8/10, purpose ("compare it to just a regular llm as a judge"). The ORM sees thousands of labeled training traces; a zero-shot judge sees none. A win against a weak baseline is the first objection any reader raises. Separately, the D5 CSAT rater is an LLM: a judge from the same family shares its biases on the CSAT target.
Plan baseline: D6 approved judge_baseline.py ("LLM-as-judge with an outcome rubric"); arms, model family and input unspecified.
Runtime evidence: upstream src/tau2/evaluator/review_llm_judge.py exists (prompt pattern to reuse). No judge outcome data yet.
Comparison grid:
| Choice | Current | A | B | C |
|---|---|---|---|---|
| R11 judge arms | one unspecified judge | zero-shot rubric + few-shot (k=8 labeled examples drawn from the train split only) | zero-shot rubric only | zero-shot + few-shot + prompt optimized on the train split (validation-selected) |
| Judge model | unspecified | current frontier model, different family from the D5 CSAT rater | same rule | same rule |
| Judge input | unspecified | identical permitted trace the ORM sees (R5 allowlist) | same | same |
| Extra API cost | n/a | ~2x zero-shot judging | baseline | ~3-5x zero-shot judging |
| R1-R10 | approved | unchanged | unchanged | unchanged |
Question D14:
D14 — How strong should the LLM-judge baseline be?
Project/branch/task: new τ²-bench telecom fork (v2.0.0) for the ORM experiment.
ELI10: If the small ORM learns from thousands of examples and the big LLM judge sees none, "ORM wins" is unsurprising and easy to dismiss. Giving the judge a few labeled examples in its prompt, from training data only, makes it a fair fight. The judge should also come from a different model company than the LLM that writes CSAT ratings, so it does not get a free match on those ratings. Both judges see exactly the same trace.
Stakes if we pick wrong: a weak baseline makes an ORM win meaningless; a same-family judge inflates the LLM side on CSAT.
Recommendation: A) Zero + few-shot because it answers the obvious "did you give the judge examples?" objection at about twice the judging cost.
Completeness: A=8/10, B=5/10, C=10/10
Pros / cons:
A) Zero + few-shot (recommended)
  ✅ Two arms show how much a handful of examples helps the LLM, a useful finding by itself
  ✅ Same input and family-separation rules for every arm keep the comparison clean
  ❌ Doubles judge API spend on the test set (human: ~1 day / CC: ~1 hr)
B) Zero-shot only
  ✅ Cheapest and simplest, matches the literal ask ("just a regular llm as a judge")
  ❌ Readers will say the judge was handicapped; the result is hard to defend
C) Zero + few + optimized
  ✅ Strongest possible baseline; an ORM win here is very convincing
  ❌ Prompt optimization adds a training loop for the judge and 3-5x judging cost (human: ~4 days / CC: ~3 hrs)
Net: a fair baseline at modest cost versus the cheapest arm or the most expensive one.
Header: Judge arms
Options:
A) Zero + few-shot (recommended)
Two judge arms: zero-shot with the outcome rubric, and few-shot with k=8 labeled examples drawn from the train split only (fixed seed, stratified by label). Judge model = a current frontier model from a different family than the D5 CSAT rater. Both arms and the ORM read the identical permitted trace (R5 allowlist) and output the same three predictions.
B) Zero-shot only
One judge arm: zero-shot with the outcome rubric. Same family-separation and identical-input rules as above.
C) Zero + few + optimized
Three arms: zero-shot, few-shot (k=8, train split only), and a prompt optimized on the train split with selection on validation. Same family-separation and identical-input rules.

State: approved
Actual answer: A) Zero + few-shot (recommended), D14 answer 2026-10-27
Accepted scope: Two judge arms: zero-shot with the outcome rubric, and few-shot with k=8 labeled examples drawn from the train split only (fixed seed, stratified by label). Judge model = a current frontier model from a different family than the D5 CSAT rater. Both arms and the ORM read the identical permitted trace (R5 allowlist) and output the same three predictions.
History: none

Section 3 dispositions: R10 accepted (D13 Suite + golden); R11 accepted (D14 Zero + few-shot); 33 proposed-path gaps carried as required proof of approved behavior (see Test Plan artifact). Eval scope: [→EVAL] CSAT rater prompt (ratings must rise with resolved tasks on a 20-trace fixture) and judge prompt (output parses into the three predictions on 20 fixtures, stable across 3 reruns).

## Review Section 4: Performance

Scale (estimate, unverified until a pilot): N customers × 9 variants × ~1 contact per customer per 3-4 virtual weeks over a 180-day horizon. Historical telecom evidence: 9 connected simulations cost $0.181 total, ~$0.02 each, model mix unknown (internal research notes docs/research/2026-09-07-tau2-continuous-insight-validation.md). Frontier agent variants will cost more per conversation.

Findings:
1. [P1] (confidence: 7/10) campaign sizing — churn within 28 days is a rare label (single-digit percent per month in typical subscription businesses). AUROC with a usable confidence interval needs on the order of 200+ positives in the test split, which implies roughly 10-15k mature traces in total. Neither the per-conversation cost, the trace length (R7 context) nor the hazard's realized churn rate is known. → R12 (D15).
2. [P2] (confidence: 7/10) world.py project_state — rebuilding a customer's state from the ledger before each conversation is O(events for that customer), at most hundreds; fine. Required: index on (customer_id, virtual_day). Necessary implementation; no question.
3. [P2] (confidence: 7/10) web search — LIKE scans over tens of thousands of transcripts are slow. Required: SQLite FTS5 index over message and tool text, built when a campaign is exported. Necessary implementation of item 6 search; no question.
4. [P2] (confidence: 6/10) Medium confidence, verify this is actually an issue. Day-wave long tail (D12) — one stuck conversation stalls a wave. Required: per-conversation step and wall-clock limits from upstream run config; a timed-out conversation is recorded as an infrastructure error (item 3, kept separate from task success), never retried blindly. Necessary implementation; no question.
5. [P2] (confidence: 6/10) Medium confidence, verify this is actually an issue. Analytics export (item 5) — provider import APIs cap batch size and rate. Required: batched import with retry/backoff after the campaign, never inline in the wave loop. Necessary implementation; no question.

### R12: Pilot gate before the full campaign
Finding: Section 4 #1, P1, confidence 7/10, reviewer plan-eng-review.
Plan baseline: item 7 approved "budgets"; campaign size and spend not specified.
Runtime evidence: none for this design. Historical ~$0.02 per telecom conversation with an unknown model mix.
Comparison grid:
| Choice | Current | A | B |
|---|---|---|---|
| R12 sizing | unspecified | pilot first: 30 customers × 9 variants × 60 virtual days, hard cap $75; measure cost per conversation, p95 trace tokens, label base rates; tune hazard until 28-day churn is 3-10% per month and repeat contact 10-30%; then compute full-run size for ≥200 test positives and set its hard cap from measured cost; preregister before the full run | fixed now: 400 customers × 9 variants × 180 virtual days with a hard budget cap chosen now |
| Hazard calibration | D5 approved, values unset | tuned on pilot only, frozen before the full run | set once from defaults, not tuned |
| R1-R11 | approved | unchanged | unchanged |
Question D15:
D15 — Run a small priced pilot before the full dataset?
Project/branch/task: new τ²-bench telecom fork (v2.0.0) for the ORM experiment.
ELI10: Nobody knows yet what one telecom conversation costs with these three agent models, how long the traces are, or how often the made-up churn model actually makes customers leave. A small pilot (30 customers, about $75 cap) measures all three. Then you tune the churn model so churn lands in a believable range, freeze it, and size the real run so the test set has enough churners to score the judges.
Stakes if we pick wrong: a full run with a badly tuned hazard can spend hundreds of dollars producing almost no churners, or churn everyone in week two, and you only find out at the end.
Recommendation: A) Pilot gate because it turns cost, trace length and label balance from guesses into measurements before the expensive run.
Completeness: A=9/10, B=5/10
Pros / cons:
A) Pilot gate (recommended)
  ✅ Measured per-conversation cost turns the full-run budget into arithmetic, not a guess
  ✅ Hazard is tuned on pilot data and frozen before the full run, so the full run cannot be tuned to the result
  ❌ Adds a pilot step and a short wait before the real dataset (human: ~1 day / CC: ~1 hr, plus ≤$75)
B) Fixed size now
  ✅ One run, no extra step; fastest calendar path to a dataset
  ✅ Budget known up front
  ❌ Untested hazard and unknown costs: high risk of a skewed or useless dataset after full spend
Net: a cheap measured pilot versus a single blind run.
Header: Pilot
Options:
A) Pilot gate (recommended)
Pilot: 30 customers × 9 variants × 60 virtual days, hard cap $75. Measure cost per conversation, p95 trace tokens (feeds R7 model choice), and label base rates. Tune hazard until 28-day churn is 3-10% per month and 7-day repeat contact 10-30%, then freeze hazard config. Size the full run for ≥200 positive churn labels in the test split; its hard budget cap = measured cost × planned conversations × 1.3. Write PREREGISTRATION.md (splits, metrics, arms, sizes, frozen config hash) before the full run starts. Pilot data is never used in the ORM experiment.
B) Fixed size now
Full run of 400 customers × 9 variants × 180 virtual days with a hard budget cap set now. Hazard uses default weights without tuning. Preregistration written before the run.

State: approved
Actual answer: A) Pilot gate (recommended), D15 answer 2026-10-27
Accepted scope: Pilot: 30 customers × 9 variants × 60 virtual days, hard cap $75. Measure cost per conversation, p95 trace tokens (feeds R7 model choice), and label base rates. Tune hazard until 28-day churn is 3-10% per month and 7-day repeat contact 10-30%, then freeze hazard config. Size the full run for ≥200 positive churn labels in the test split; its hard budget cap = measured cost × planned conversations × 1.3. Write PREREGISTRATION.md (splits, metrics, arms, sizes, frozen config hash) before the full run starts. Pilot data is never used in the ORM experiment.
History: none

Section 4 dispositions: #1 accepted (D15 Pilot gate); #2-#5 accepted as necessary implementation of approved items 3, 5, 6 and D12.

## Outside Voice (Codex, gpt-6-astra, plan-review tier frontier, 2026-10-27)

Run from an empty scratch git repo (no project repo exists); input = condensed working plan (original spec + accepted D1-D15 + section findings, 22.7KB). OUTSIDE_STATUS: completed. 10 findings (7 High, 3 Medium), recommendation: revise the experimental target and state-transition contract before implementation.

Cross-model tension, dispositions:
- OV1 (High) D7 window labels measure risk under the continuation policy, not causal effect; logged hazard terms are not counterfactual attribution. → Factual correction accepted: D7's oracle quantity is renamed "direct hazard contribution" (the trace's own term at its checkpoint under the authored model), and no writeup may call it total causal effect. Claim scope → R13 (D16).
- OV2 (High) ORM can win by recovering the one authored hazard formula. → R13 (D16).
- OV3 (High) history vs template init precedence; campaign-wide preflight cannot validate future states. → Correction to D11 implementation, same approved contract ("pairs that error are skipped and logged, never run"): precedence = customer history first, template init actions second (the task's issue overrides only the fields it sets); the gold-replay applicability check runs at scheduling time against the customer branch's actual projected state, and a failing template is replaced by the next eligible template for that customer (logged). No question.
- OV4 (High) variant isolation missing. → Correction accepted as necessary implementation of item 7 and item 1: every state row, event, checkpoint, oracle row and export carries branch_id = (population_customer_id, variant_id); population_customer_id is used for matching and for the D6 customer-level split (all 9 branches of a customer land in the same split). Index (branch_id, virtual_day) replaces (customer_id, virtual_day). No question.
- OV5 (High) day waves can run two conversations for one branch on the same day. → Correction accepted as necessary implementation of D12's ordering contract: at most one conversation per branch per virtual day; a second due contact moves to the next day (logged as deferred_contact). No question.
- OV6 (High) 200 positive rows are not 200 independent outcomes. → R14.
- OV7 (High) resume is not atomic past upstream completion. → Correction accepted as necessary implementation of item 7 cancel/resume and D12: each conversation's post-processing (result insert, CSAT, hazard draw, actions, oracle rows) commits in one SQLite transaction keyed by a deterministic checkpoint id (branch_id, virtual_day, conversation_seq); replays are no-ops; a branch with a pending CSAT does not advance past that day until resolved. No question.
- OV8 (Medium) pilot never tests predictive signal. → R15.
- OV9 (Medium) target and scoring contracts disagree. → Factual correction: target schema frozen to the three D10 heads (churn-or-downgrade ≤28d, repeat contact ≤7d, CSAT); "renewal" is dropped from score.py as a separate metric (renewal = no churn-or-downgrade at term end, already covered). Primary comparison criterion → R16.
- OV10 (Medium) budget cap checked after the whole wave. → Correction accepted as necessary implementation of D15's "hard cap": admission control reserves a per-conversation cost bound (agent + simulated user + CSAT + retries, from pilot p95 cost) before submitting each conversation; a wave admits only what fits the remaining budget. No question.

### R13: What the experiment is allowed to claim
Finding: OV1 + OV2, High, confidence 8/10, reviewer Codex (outside voice), confirmed by plan-eng-review. Train and test share one frozen hazard generator and one set of agent policies, so an ORM can win by recovering that formula.
Plan baseline: D5 (hybrid hazard), D7 (window + oracle), D15 (hazard frozen after pilot). Claim scope unspecified.
Runtime evidence: none (design-level).
Comparison grid:
| Choice | Current | A Apply | B Keep | C Investigate | D Defer |
|---|---|---|---|---|---|
| R13 generalization test | none; one generator, one policy set | add a held-out test set generated with 2 perturbed hazard configs (weights shifted, one extra trait term) and 1 held-out agent model not in training; claim stated as "prediction within this simulator family", direct hazard contribution never called causal | unchanged; claim unstated | measure on pilot whether a logistic model fit to config 1 degrades on a perturbed config, then return to this row | no change now; row stays open |
| D5, D7, D15 | approved | unchanged | unchanged | unchanged | unchanged |
| R14-R16 | pending | pending | pending | pending | pending |
Question D16:
D16 — Add a held-out simulator test so the ORM can't win by memorizing one formula?
Project/branch/task: new τ²-bench telecom fork (v2.0.0) for the ORM experiment.
ELI10: The churn rules are written by you. If training and test data both come from exactly the same rules, the ORM might just learn those rules, which says nothing about real customers. Codex flagged this. The fix: also test on data made with slightly different rules and with an agent model the ORM never saw. If the ORM still beats the LLM judge there, the result means more. And the writeup should say "predicts outcomes in this simulator", not "measures causal impact".
Stakes if we pick wrong: a reviewer dismisses the result as "the small model learned your formula", and you cannot answer.
Recommendation: A) Apply because a held-out generator is the only evidence against formula-memorizing, and its cost is one extra small campaign.
Note: options differ in kind, not coverage — no completeness score.
Pros / cons:
A) Apply (recommended)
  ✅ Shows whether the ORM's edge survives rule changes and an unseen agent, the strongest defensible claim
  ✅ Claim wording fixed up front, so the writeup cannot drift into causal language
  ❌ One more campaign (about 15-20% of full-run spend) and a second scoring table (human: ~2 days / CC: ~1.5 hrs)
B) Keep
  ✅ No extra campaign or spend
  ✅ Simplest experiment to explain
  ❌ Result cannot rule out formula recovery; claim scope left implicit
C) Investigate
  ✅ Pilot data tells you whether formula-memorizing is even a measurable risk before paying for it
  ❌ Delays the decision until after the pilot; row stays open until then
D) Defer
  ✅ Keeps current scope moving
  ❌ The gap stays open and must be resolved before any writeup
Net: one extra held-out campaign for a defensible claim versus a cheaper but easily dismissed result.
Header: Held-out sim
Options:
A) Apply (recommended)
Add a held-out evaluation campaign after the full run: same population size as the test split, generated with 2 perturbed hazard configs (weights shifted ±30%, one extra hidden-trait term) and 1 agent model not used in training. score.py reports in-distribution and held-out results side by side. PREREGISTRATION.md states the claim as prediction within this simulator family and defines "direct hazard contribution"; no causal language.
B) Keep
No held-out generator or agent; single in-distribution test split. Claim scope not written down.
C) Investigate
On pilot data only: fit the observable-feature logistic baseline on the default config, score it on a perturbed config, report the drop. Then return to this row with that number. No held-out campaign approved yet.
D) Defer
Leave this proposal unresolved; it must be decided before PREREGISTRATION.md is written.

State: approved
Actual answer: A) Apply (recommended), D16 answer 2026-10-27
Accepted scope: Add a held-out evaluation campaign after the full run: same population size as the test split, generated with 2 perturbed hazard configs (weights shifted ±30%, one extra hidden-trait term) and 1 agent model not used in training. score.py reports in-distribution and held-out results side by side. PREREGISTRATION.md states the claim as prediction within this simulator family and defines "direct hazard contribution"; no causal language.
History: none

### R14: Counting independent evidence when sizing and scoring
Finding: OV6, High, confidence 8/10, reviewer Codex (outside voice), confirmed by plan-eng-review. One churn event labels several overlapping trace windows, and 9 variants reuse each population customer, so "≥200 positive labels" (D15) can be far fewer independent outcomes, and per-row confidence intervals would be too narrow.
Plan baseline: D15 approved "Size the full run for ≥200 positive churn labels in the test split".
Runtime evidence: none (design-level).
Comparison grid:
| Choice | Current | A Apply | B Keep | C Investigate | D Defer |
|---|---|---|---|---|---|
| R14 sizing unit | ≥200 positive churn labels (rows) in test split (D15) | ≥200 distinct churn-or-downgrade events in the test split, one per branch event, however many trace windows it labels | ≥200 positive rows (unchanged) | measure on pilot how many rows each event labels and the between-variant correlation, then return to this row | unchanged now; row stays open |
| R14 uncertainty | unspecified | every metric and every ORM-vs-judge difference reported with a paired bootstrap over population customers (all 9 branches resampled together), 95% CI | unspecified | unspecified | unspecified |
| D15 other terms, D16 | approved | unchanged | unchanged | unchanged | unchanged |
| R15, R16 | pending | pending | pending | pending | pending |
Question D17:
D17 — Size and score the experiment by independent customers, not by rows?
Project/branch/task: new τ²-bench telecom fork (v2.0.0) for the ORM experiment.
ELI10: When one customer cancels, every conversation in the 28 days before it gets a "churned" label, so one cancellation can look like five. The same customer also appears in all 9 agent variants. Counting rows overstates how much evidence you have, so the "ORM beats judge" error bars come out too narrow. The fix: count actual cancellations when sizing the run, and compute error bars by resampling whole customers.
Stakes if we pick wrong: the result looks statistically significant when it is not, and the full run may be too small.
Recommendation: A) Apply because it is the correct statistic for clustered data and costs only a bigger run if the pilot shows heavy overlap.
Note: options differ in kind, not coverage — no completeness score.
Pros / cons:
A) Apply (recommended)
  ✅ Error bars reflect real independent evidence, so a reported win holds up to review
  ✅ Paired bootstrap compares ORM and judge on the same customers, which tightens honest intervals
  ❌ Full run likely grows (exact size from the pilot), raising spend under the D15 cap formula (human: ~1 day / CC: ~45 min)
B) Keep
  ✅ Smaller full run and simpler scoring code
  ❌ Overstated confidence; a careful reader can overturn the conclusion
C) Investigate
  ✅ Pilot measures how much windows overlap before committing to a larger run
  ❌ Leaves sizing open until after the pilot
D) Defer
  ✅ Nothing changes now
  ❌ Must be resolved before PREREGISTRATION.md, or the full run is sized wrong
Net: honest error bars and possibly a bigger run versus overstated confidence.
Header: Sizing unit
Options:
A) Apply (recommended)
Replace D15's sizing rule with: ≥200 distinct churn-or-downgrade events in the test split (one per branch event, however many trace windows it labels). Every metric and ORM-vs-judge difference is reported with a paired bootstrap over population customers (all 9 branches of a customer resampled together), 95% CI. All other D15 terms unchanged.
B) Keep
Keep D15's rule of ≥200 positive churn labels (rows) in the test split; uncertainty method unspecified.
C) Investigate
On pilot data, measure rows per churn event and between-variant correlation, then return to this row with those numbers. D15's rule stays as approved meanwhile.
D) Defer
Leave this proposal unresolved; it must be decided before PREREGISTRATION.md is written.

State: approved
Actual answer: A) Apply (recommended), D17 answer 2026-10-27
Accepted scope: Replace D15's sizing rule with: ≥200 distinct churn-or-downgrade events in the test split (one per branch event, however many trace windows it labels). Every metric and ORM-vs-judge difference is reported with a paired bootstrap over population customers (all 9 branches of a customer resampled together), 95% CI. All other D15 terms unchanged.
History: D15 sizing rule "≥200 positive churn labels in the test split" superseded by D17.

### R15: Pilot must prove traces carry predictive signal
Finding: OV8, Medium, confidence 7/10, reviewer Codex (outside voice), confirmed by plan-eng-review. D15's pilot checks cost, trace length and label rates. All three can pass while hidden traits and noise dominate outcomes, leaving nothing in the trace for either judge to learn.
Plan baseline: D15 approved pilot gate (cost, p95 tokens, base rates, hazard tuning to rate bands).
Runtime evidence: none (design-level).
Comparison grid:
| Choice | Current | A Apply | B Keep | C Investigate | D Defer |
|---|---|---|---|---|---|
| R15 signal gate | none | pilot also must show (1) observable-feature logistic baseline AUROC on churn-or-downgrade with customer-cluster bootstrap 95% CI lower bound > 0.55, and (2) a churn-or-downgrade rate difference between best and worst agent variant whose 95% CI excludes 0; if either fails, raise the trace-signal hazard weights and re-pilot (max 2 re-pilots, each ≤$75) before the full run | no signal gate | run the pilot first, report these two numbers, then return to this row | unchanged; row stays open |
| D15, D17 | approved | unchanged | unchanged | unchanged | unchanged |
| R16 | pending | pending | pending | pending | pending |
Question D18:
D18 — Should the pilot also prove the traces actually predict churn?
Project/branch/task: new τ²-bench telecom fork (v2.0.0) for the ORM experiment.
ELI10: The pilot as planned checks cost and churn rates. But churn could be driven almost entirely by hidden traits and luck, so nothing in the conversation predicts it, and neither the ORM nor the LLM judge could beat random. Two cheap checks catch that: a simple model on visible facts (task solved? tool errors? repeat calls?) must beat random, and better agents must actually produce less churn. If either check fails, turn up how much the conversation matters in the churn rules and re-pilot.
Stakes if we pick wrong: the full run spends the real budget on a dataset where every judge scores near random, and the experiment says nothing.
Recommendation: A) Apply because it costs only analysis on pilot data you are already collecting, and it protects the full-run spend.
Note: options differ in kind, not coverage — no completeness score.
Pros / cons:
A) Apply (recommended)
  ✅ Guarantees the full dataset has learnable signal before the expensive run
  ✅ Confirms agent quality changes outcomes, the premise of the whole experiment
  ❌ Up to 2 re-pilots (≤$150 extra) if the first pilot fails the gate (human: ~1 day / CC: ~45 min)
B) Keep
  ✅ Fastest path from pilot to full run
  ❌ Risk of a full run where no judge beats random
C) Investigate
  ✅ See the two numbers before deciding thresholds
  ❌ Threshold set after seeing data, which weakens the preregistration
D) Defer
  ✅ Nothing changes now
  ❌ Must be resolved before the full run starts
Net: a cheap learnability check versus risking the full budget on an unlearnable dataset.
Header: Signal gate
Options:
A) Apply (recommended)
Add to the D15 pilot gate: (1) a logistic regression on observable features (native task success, tool error count, turns, prior repeat contacts) predicts churn-or-downgrade with AUROC whose customer-cluster bootstrap 95% CI lower bound is > 0.55; (2) the churn-or-downgrade rate difference between best and worst agent variant has a 95% CI excluding 0. If either fails, raise the trace-signal hazard weights and re-pilot (max 2 re-pilots, each ≤$75). Thresholds go into PREREGISTRATION.md before the pilot runs.
B) Keep
Pilot gate stays as D15 approved: cost, p95 tokens, base rates and hazard tuning only.
C) Investigate
Run the D15 pilot, report the two numbers, then return to this row to set thresholds.
D) Defer
Leave this proposal unresolved; it must be decided before the full run starts.

State: approved
Actual answer: A) Apply (recommended). Adi, 2026-10-27 chat: "take reccomended and then i'll review after".
Accepted scope: Add to the D15 pilot gate: (1) a logistic regression on observable features (native task success, tool error count, turns, prior repeat contacts) predicts churn-or-downgrade with AUROC whose customer-cluster bootstrap 95% CI lower bound is > 0.55; (2) the churn-or-downgrade rate difference between best and worst agent variant has a 95% CI excluding 0. If either fails, raise the trace-signal hazard weights and re-pilot (max 2 re-pilots, each ≤$75). Thresholds go into PREREGISTRATION.md before the pilot runs.
History: D18 brief sent once; Adi replied by message instead of selecting, authorizing recommended options.

### R16: Primary comparison criterion
Finding: OV9, Medium, confidence 8/10, reviewer Codex (outside voice). The plan lists many metrics but no single preregistered criterion that decides "ORM wins / judge wins / inconclusive".
Plan baseline: D6 score.py metric list; D10 heads; D17 cluster bootstrap.
Comparison grid:
| Choice | Current | A Apply (recommended) | B Keep |
|---|---|---|---|
| R16 primary criterion | none | in-distribution test split: AUROC(ORM) − AUROC(best LLM-judge arm, arm chosen on validation) for churn-or-downgrade ≤28d, paired customer-cluster bootstrap 95% CI. Lower bound > 0 = ORM wins; upper bound < 0 = judge wins; otherwise inconclusive. Secondary (reported, not deciding): same difference on the D16 held-out set, Brier, repeat-contact AUROC, CSAT Spearman, top-10% capture of traces ranked by direct hazard contribution | many metrics, no decision rule |
State: approved
Actual answer: A) Apply (recommended), auto-decided under Adi's 2026-10-27 instruction "take reccomended"; no brief was shown for this row.
Accepted scope: column A above, written into PREREGISTRATION.md before the pilot.
History: none

Approval readiness: PASS — checked R1 (D5), R2 (D6), R3 (D7), R4 (D8), R5 (spec item 7), R6 (D9), R7 (D10), R8 (D11), R9 (D12), R10 (D13), R11 (D14), R12 (D15, sizing superseded by D17), R13 (D16), R14 (D17), R15 (D18, recommended per Adi's instruction), R16 (recommended per Adi's instruction); scope D1-D4; outside-voice corrections OV1, OV3, OV4, OV5, OV7, OV9, OV10 recorded as necessary implementation of approved contracts.

## NOT in scope

- Langfuse/OpenTelemetry upload (item 4): deferred (D3). The ORM reads local traces, which already carry per-call role tags.
- Retail or airline domains: deferred (D9). Telecom is the only domain with native subscription state.
- Counterfactual paired re-runs as causal labels: not chosen (D7). Grading uses direct hazard contribution instead.
- Prompt-optimized judge arm: not chosen (D14). Zero-shot and 8-shot arms only.
- Real-customer transfer claims: excluded (D16). The claim is limited to prediction within this simulator family.
- ChurnBench code reuse: excluded by Adi (decision log 6918b02a). Its patterns may be re-implemented natively.

## What already exists

- **Upstream τ²-bench v2.0.0 (reused, not rebuilt):**
  - `runner/batch.py` + `runner/checkpoint.py`: batch execution, resume, single writer (D12).
  - `evaluator/evaluator_env.py`: DB-hash gold-replay grading, which stays the native task-success measure (D13).
  - `evaluator/review_llm_judge.py`: prompt pattern for the judge baseline (D14).
  - `domains/telecom/environment.py:97`: `get_environment(db=...)` injection.
  - Telecom `Customer`/`Plan`/`Line`/`Bill` models and the FastAPI dependency.
- **ChurnBench (an earlier internal project (ChurnBench)):** partial versions of all 7 items in a CRM domain. Not reused, by decision. Patterns worth re-implementing natively:
  - semantic-key RNG (`randomness.py:19 EventKey`)
  - allowlisted training projection (`contracts/optimizer.py:154 TrainingBundle`)
  - Mixpanel import shape (`benchloop/product_events.py:56`)
- **Rafit's tau2 testbed (an earlier private tau2 testbed):** it has a workload mode with a ledger and virtual time. Adi's GitHub account gets a 404 on it, so it is not reused.

## Failure modes

| Path | Realistic failure | Handling / test | Visible? |
|---|---|---|---|
| Retarget (D11) | template ID left in free text | ID-map substitution + no-residual-ID assert + equivalence test (D13) | test fails loudly |
| History replay | history conflicts with task init | precedence rule + per-conversation applicability check, template swapped and logged (OV3) | logged swap |
| Day wave | one conversation hangs | upstream step/time limits → infrastructure error, kept separate (Sec 4 #4) | infra-error KPI |
| CSAT LLM call | provider error | retry, then branch pending, never defaulted (Sec 1 #4, OV7) | pending count in UI/CLI |
| Crash mid-post-processing | duplicate transitions | one transaction per checkpoint id, replay no-op (OV7) | none needed; idempotent |
| Budget | wave overshoots | admission reservation per conversation (OV10) | stop message at day boundary |
| Export | private field leaks | allowlist + sentinel leakage test (R5) | test fails loudly |
| Analytics sink | 4xx/5xx/rate limit | batched retry, ledger untouched (Sec 4 #5) | CLI error summary |
| Pilot | no learnable signal | D18 gate blocks the full run | gate prints FAIL |

Critical gaps (no test, no handling, silent): 0.

## Worktree parallelization strategy

| Step | Modules touched | Depends on |
|---|---|---|
| P0 fork + baseline | repo root | — |
| P1 world + retarget + ledger schema | src/tau2/lifecycle/ (world), tests/lifecycle/ | P0 |
| P2-P4 checkpoint, kpis, campaign, export | src/tau2/lifecycle/ | P1 |
| P5 web UI | web/, tests/lifecycle/test_web.py | P1 (ledger schema) |
| P6 experiment scaffolding | experiments/orm/, tests/orm/ | P1 (schema); export row format from P4 (use fixture until then) |
| P7-P8 pilot, full run, scoring | experiments/orm/ (runs) | P2-P6 |

Parallel lanes: Lane A: P2 → P3 → P4 (shared lifecycle/). Lane B: P5 (web/). Lane C: P6 (experiments/orm/).
Execution order: P0 → P1 sequentially. Then launch A, B and C together and merge all three. Then P7 → P8.
Conflict flags: the ledger schema in world.py is shared by all lanes. Freeze it at the end of P1; any later schema change goes through Lane A and is rebased into B and C.

## Implementation Tasks
Built from this review's findings; each task traces to one finding above. Ratio assumption: features ~30x, tests ~50x, research ~3x.

- [ ] **T1 (P1, human: ~2h / CC: ~10min)** — repo — Fork tau2-bench v2.0.0 into a new private repo and record the upstream test baseline
  - Surfaced by: Scope record + R6 (D9 pin)
  - Files: repo root
  - Verify: `uv run pytest -q` green
- [ ] **T2 (P1, human: ~4d / CC: ~3h)** — lifecycle/world.py — Ledger with branch_id, projection, synthetic customers, retarget + history replay, applicability check
  - Surfaced by: Section 1 #1 (R8/D11), OV3, OV4, Section 2 #1
  - Files: src/tau2/lifecycle/world.py, tests/lifecycle/test_world.py, tests/lifecycle/test_retarget_equivalence.py
  - Verify: `uv run pytest tests/lifecycle -q`
- [ ] **T3 (P1, human: ~3d / CC: ~3h)** — lifecycle/checkpoint.py + kpis.py — Hybrid hazard, CSAT call (pending on failure), action validation, oracle rows, window_labels + censoring
  - Surfaced by: R1/D5, R3/D7, Section 1 #4, Section 2 #2, #3, OV9
  - Files: src/tau2/lifecycle/checkpoint.py, src/tau2/lifecycle/kpis.py, tests/lifecycle/
  - Verify: `uv run pytest tests/lifecycle -q`
- [ ] **T4 (P1, human: ~4d / CC: ~3h)** — lifecycle/campaign.py + cli.py — Day waves, 1 conversation/branch/day, budget admission, atomic resume, config hash, keyed RNG, failure injection
  - Surfaced by: R9/D12, R4/D8, OV5, OV7, OV10, Section 2 #4, #5
  - Files: src/tau2/lifecycle/campaign.py, src/tau2/lifecycle/cli.py, tests/lifecycle/test_campaign_e2e.py
  - Verify: `uv run pytest tests/lifecycle/test_campaign_e2e.py -q`
- [ ] **T5 (P1, human: ~2d / CC: ~1.5h)** — lifecycle/export.py — Allowlist training export with sentinel leakage test; Mixpanel/Amplitude batched sink
  - Surfaced by: R5, Section 1 #5, Section 4 #5, D1
  - Files: src/tau2/lifecycle/export.py, tests/lifecycle/test_export.py
  - Verify: `uv run pytest tests/lifecycle/test_export.py -q`
- [ ] **T6 (P2, human: ~2w / CC: ~6h)** — web/ — Cohort dashboard, FTS5 search, customer state, timelines, side-by-side compare
  - Surfaced by: D2, Section 4 #3
  - Files: web/server.py, web/index.html, web/app.js, tests/lifecycle/test_web.py
  - Verify: `uv run pytest tests/lifecycle/test_web.py -q` + browser smoke
- [ ] **T7 (P1, human: ~1w / CC: ~5h)** — experiments/orm/ — PREREGISTRATION.md, customer-level split, judge arms, LoRA ORM, score.py with the R16 rule and cluster bootstrap
  - Surfaced by: R2/D6, R7/D10, R11/D14, R14/D17, R16, R13/D16
  - Files: experiments/orm/*, tests/orm/
  - Verify: `uv run pytest tests/orm -q` on toy data
- [ ] **T8 (P1, human: ~2d / CC: ~1h + ≤$75-225)** — pilot — Run the pilot, apply the D15 + D18 gates, freeze hazard config, size the full run per D17
  - Surfaced by: R12/D15, R15/D18, R14/D17
  - Files: experiments/orm/pilot_gate.py, campaign configs
  - Verify: `pilot_gate.py` prints PASS
- [ ] **T9 (P1, human: ~1w / CC: ~3h + run spend)** — full run — Full campaign + held-out campaign, judges, ORM training, scoring
  - Surfaced by: R2/D6, R13/D16, R16
  - Files: experiments/orm/ outputs
  - Verify: `score.py` prints the R16 verdict with 95% CIs

## TODOS (seed for the new repo's TODOS.md; auto-decided "A) Add" under Adi's "take recommended")

- **Langfuse/OTel trace upload**
  - What: a post-run uploader with agent / simulated-user / lifecycle span roles.
  - Why: Virgo/Langfuse demos of these runs.
  - Pros: product demo.
  - Cons: network dependency and credentials.
  - Context: deferred by D3; per-call role tags already exist in the local traces.
  - Depends on: T4.
- **Retail domain**
  - What: a second domain with an authored membership subscription.
  - Why: a cross-domain generalization test.
  - Pros: a stronger claim.
  - Cons: a second hazard calibration.
  - Context: deferred by D9.
  - Depends on: T9 result.

## Unresolved decisions

None. D18 and R16 were set to the recommended option on Adi's instruction "take reccomended and then i'll review after". Adi's review of this file is the remaining approval step.

## Completion summary

- Step 0: Scope Challenge — scope reduced per recommendation (item 4 deferred; experiment harness added)
- Reality rows: N/A (plan target; upstream interfaces verified via GitHub API: telecom env, evaluator, batch runner, tasks.json)
- Architecture Review: 5 issues found
- Code Quality Review: 5 issues found
- Test Review: diagram produced, 33 gaps identified (all proposed paths; repo not created)
- Performance Review: 5 issues found
- NOT in scope: written
- What already exists: written
- TODOS.md updates: 2 items proposed (auto-decided Add; written in this plan because the repo does not exist yet)
- Failure modes: 0 critical gaps flagged
- Unresolved decisions: 0 in this review
- Outside voice: Codex (gpt-6-astra), completed, 10 findings (7 High, 3 Medium); all resolved (6 as corrections, 4 via D16-D18 + R16)
- Parallelization: 3 lanes, 3 parallel / 2 sequential phases (P0-P1 before, P7-P8 after)
- Lake Score: 3/12 (D1, D2, D13 picked the 10/10 option)

Artifacts: Test Plan at `(local review artifact)`.

### Suppressed findings (appendix)

- (confidence: 4/10) Telecom dual-control conversations may run much longer than retail. Unmeasured until the pilot; D10's model choice already waits on measured p95 trace tokens.

## Design Review: Showcase UI (plan-design-review, 2026-10-27)

Target: item 6 / D2 (web/). No DESIGN.md exists in the new repo. Reference: Adi's lab-report style in an earlier internal project's design system (ChurnBench) (tokens only; no code reused). Upstream `web/leaderboard` is a separate React leaderboard site, not reused.

Mockups: none. The gstack designer started, but all 3 variants of the customer-timeline screen failed with OpenAI image API 429 (rate limited); 0/3 saved, no board. Per the skill rule the review continued text-only.

Decision mode: Adi chose D3 "Auto-take recommended". Every decision below was auto-taken, is labeled with its pass id in "Showcase UI spec", and is reversible by editing that section.

### Outside voices

- Codex (gpt-6-astra, design voice): completed. Classified the surface as APP UI. Findings: 2 Critical, 5 High, 1 Medium. Recommendation: revise item 6 before implementation.
  - Critical: outcome maturity is missing from the display; no evidence/attribution model.
  - High: first-screen hierarchy; comparison alignment; temporal customer identity; unadopted visual identity; smoke tests that don't prove usability.
  - Medium: search ends at a conversation instead of the matching evidence.
- Claude subagent (design completeness): completed. Findings: 3 Critical, 4 High, 2 Medium.
  - Critical: no defined user; screen order buries the point; private-data visibility unspecified.
  - High: "connect" reads as "caused"; missing states; compare under-defined; generic patterns.
  - Medium: transcript is a dead end; scope creep.

```
DESIGN OUTSIDE VOICES — LITMUS SCORECARD (plan as submitted):
  Check                                    Claude  Codex  Consensus
  1. Brand unmistakable in first screen?   NO      NO     CONFIRMED NO
  2. One strong visual anchor?             NO      NO     CONFIRMED NO
  3. Scannable by headlines only?          NO      NO     CONFIRMED NO
  4. Each section has one job?             NO      NO     CONFIRMED NO
  5. Cards actually necessary?             NO      NO     CONFIRMED NO (good: use tables)
  6. Motion improves hierarchy?            NO      NO     CONFIRMED (no motion needed)
  7. Premium without decorative shadows?   NO      NO     CONFIRMED NO (unspecified)
  Hard rejections triggered:               at risk #1, #7 (both)   CONFIRMED risk
```

After the spec: (1) yes, product line + experiment line; (2) yes, the 3×3 matrix; (3) yes, section headings state each area's job; (4) yes; (5) no cards; (6) motion limited to the divergence scroll; (7) yes, no shadows by spec. The hard-rejection risk is resolved by the table-led spec (4A).

### Passes (before → after)

| Pass | Before | After | Decisions (auto-taken, recommended) | Remaining gap |
|---|---|---|---|---|
| 1 Information architecture | 2 | 9 | 1A primary user = viewer of the experiment, researcher secondary; 1B Overview order with 3×3 matrix anchor + "Compare matched customers" primary action; 1C 4 routes, URL-held state, persistent top bar; 1D customer page "as of virtual day" context | no mockup to confirm visual weight |
| 2 Interaction states | 1 | 9 | 2A full state table incl. immature window, CSAT pending, running, budget-stopped, infra error, template swap, churn terminal, schema mismatch, 3 distinct search empties | no mockup |
| 3 Journey & arc | 2 | 8 | 3A evidence chain with event ids and "within 28 days of" wording; 3B outcome strip above every transcript | 5-year reflective layer N/A for a research tool |
| 4 AI slop risk | 3 | 8 | 4A OPERATE classification, table-led, explicit timeline glyph encoding and column list; 4B no motion except divergence scroll | unverified visually (mockups failed) |
| 5 Design system | 0 | 8 | 5A web/DESIGN.md with tokens, IBM Plex Sans + Plex Mono self-hosted, no shadows, variants by label + pattern | full design-consultation not run |
| 6 Responsive & a11y | 0 | 8 | 6A breakpoints ≥1280 / 768-1279 / <768 with horizontal scroll, never stacked compare; 6B landmarks, captions, timeline event table, keyboard model, focus ring, contrast, 44px targets | contrast of --quiet must be measured at build |
| 7 Unresolved decisions | — | — | 7A simulator-truth layer off by default (`serve --oracle`), separate endpoints, leakage test on default API; 7B manual refresh + "data as of" while running; 7C compare defaults (same failure rate, two models; any 2 of 9 with factor note) | none |

Storyboard (3A, 3B):

```
STEP | USER DOES                          | USER FEELS              | PLAN SPECIFIES?
1    | opens /, reads the one-line intro  | oriented in 5 seconds    | yes: product + experiment line, status
2    | scans the 3×3 matrix               | "worse agents, more churn"| yes: matrix with CI + n mature
3    | clicks Compare matched customers   | curious: why?            | yes: featured pair, divergence marked
4    | sees day 37: A failed, B solved    | "that's the moment"      | yes: divergence label, glyphs
5    | opens the A conversation           | sees the tool timeout    | yes: transcript + collapsed tool calls
6    | reads the outcome strip            | sees what followed, not a blame claim | yes: "within 28 days of" chain
7    | (researcher) searches refuel_data  | finds every instance     | yes: exact-anchor results, kept context
8    | (Adi, --oracle) opens Simulator truth | trusts the labels     | yes: hatched, separately labeled layer
```

### NOT in scope (design)

- Visual mockups: blocked by API rate limit. Retry offered as a TODO below.
- Dark mode: not specified; the use scene is a laptop on a desk in daylight, so light only.
- Live auto-refresh: manual refresh only (7B), since polling a live WAL ledger adds complexity for little gain.
- Marketing/landing page for the experiment: out of scope; this is a tool surface.

### What already exists (design)

- an earlier internal project's design system (ChurnBench): lab-report tokens (paper, ink, line, signal green). Re-declared in web/DESIGN.md; fonts changed to the IBM Plex superfamily.
- an earlier internal project's dashboard (ChurnBench): a prior stdlib dashboard with per-user threads and tool traces. Pattern reference only; no code reused, per the base-repo decision.
- Upstream `web/leaderboard` (React + Playwright): not reused. Its e2e setup is a reference for the browser smoke test.

### TODOS (design; auto-decided "A) Add")

- **Generate UI mockups once the image API rate limit clears**
  - What: run /design-shotgun on Overview and Compare from this spec.
  - Why: visual confirmation of hierarchy before building DT2-DT4.
  - Pros: catches visual-weight problems early.
  - Cons: a few minutes and image API spend.
  - Context: 0/3 variants saved on 2026-10-27 (429).
  - Depends on: nothing; best before DT2.

### Implementation Tasks (design review)
Refines eng task T6. Ratio assumption: features ~30x, tests ~50x.

- [ ] **DT1 (P1, human: ~4h / CC: ~15min)** — web/DESIGN.md — Write tokens, self-host IBM Plex Sans/Mono, theme browser surfaces
  - Surfaced by: Pass 5 (5A)
  - Files: web/DESIGN.md, web/fonts/, web/index.html
  - Verify: fonts load offline; computed styles use the tokens
- [ ] **DT2 (P1, human: ~3d / CC: ~1.5h)** — Overview — 3×3 matrix with CI + n mature, variant table, featured comparison, campaign status banner
  - Surfaced by: Pass 1 (1B), Pass 2 (2A)
  - Files: web/server.py, web/app.js, web/index.html
  - Verify: `uv run pytest tests/lifecycle/test_web.py -q` on the UI fixture ledger
- [ ] **DT3 (P1, human: ~4d / CC: ~2h)** — Customer page — as-of state table with diffs, timeline encoding, outcome strip, equivalent event table
  - Surfaced by: Pass 1 (1D), Pass 3 (3A, 3B), Pass 4 (4A), Pass 6 (6B)
  - Files: web/app.js, web/server.py
  - Verify: test_web.py + keyboard walk of the event table
- [ ] **DT4 (P1, human: ~2d / CC: ~1h)** — Compare — shared axis, divergence detection and label, factor-differ note, explicit gaps
  - Surfaced by: Pass 7 (7C)
  - Files: web/app.js, web/server.py
  - Verify: fixture with divergent branches opens at the right day
- [ ] **DT5 (P2, human: ~1d / CC: ~45min)** — Search — excerpt + context, exact anchor links, preserved state, 3 distinct empty/error states, pagination
  - Surfaced by: Pass 1 (1C), Pass 2 (2A)
  - Files: web/server.py, web/app.js
  - Verify: test_web.py search cases
- [ ] **DT6 (P1, human: ~1d / CC: ~45min)** — Simulator truth layer — `serve --oracle`, separate /api/oracle/*, hatched panel; extend the R5 sentinel test to the default API
  - Surfaced by: Pass 7 (7A), R5
  - Files: src/tau2/lifecycle/cli.py, web/server.py, tests/lifecycle/test_export.py
  - Verify: the sentinel never appears in default API responses
- [ ] **DT7 (P1, human: ~2d / CC: ~1.5h)** — UI fixture ledger + states + a11y checks (contrast of --fail/--quiet, focus, landmarks, captions)
  - Surfaced by: Pass 2 (2A), Pass 6 (6A, 6B)
  - Files: tests/lifecycle/fixtures/ui_ledger.sqlite, tests/lifecycle/test_web.py
  - Verify: browser smoke at 1440 / 1024 / 390 widths

## Design Review Completion Summary

```
  +====================================================================+
  |         DESIGN PLAN REVIEW — COMPLETION SUMMARY                    |
  +====================================================================+
  | System Audit         | no DESIGN.md; UI scope = 4 routes (web/)     |
  | Step 0               | 2/10 initial; focus: all 7 passes            |
  | Pass 1  (Info Arch)  | 2/10 → 9/10 after fixes                     |
  | Pass 2  (States)     | 1/10 → 9/10 after fixes                     |
  | Pass 3  (Journey)    | 2/10 → 8/10 after fixes                     |
  | Pass 4  (AI Slop)    | 3/10 → 8/10 after fixes                     |
  | Pass 5  (Design Sys) | 0/10 → 8/10 after fixes                     |
  | Pass 6  (Responsive) | 0/10 → 8/10 after fixes                     |
  | Pass 7  (Decisions)  | 3 resolved, 0 deferred                      |
  +--------------------------------------------------------------------+
  | NOT in scope         | written (4 items)                           |
  | What already exists  | written                                     |
  | TODOS.md updates     | 1 item proposed (auto-added)                |
  | Approved Mockups     | 0 generated (429), 0 approved               |
  | Decisions made       | 15 added to plan (auto-taken per D3)        |
  | Decisions deferred   | 0                                           |
  | Overall design score | 0/10 → 8/10                                 |
  +====================================================================+
```

Unresolved design decisions: none. All 15 were auto-taken under Adi's D3 instruction and await his read of "Showcase UI spec".

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Outside Review | Codex via `/plan-eng-review` + `/plan-design-review` outside voices | Independent 2nd opinion | 2 | completed (issues_found) | eng: 10 findings, resolved; design: 8 findings, resolved |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES OPEN (mapped to T1-T9) | 48 issues, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 1 | CLEAR (PLAN) | score: 0/10 → 8/10, 15 decisions |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **OUTSIDE COVERAGE:** codex (openai, gpt-6-astra): plan-review phase completed with 10 findings; design phase completed with 8 findings. Claude design subagent (in-host) completed with 9 findings; this is native, not outside coverage. Mockup generation unavailable (OpenAI image 429).
- **CROSS-MODEL:** On the design phase, Codex and the Claude subagent independently agreed on 5 of 7 litmus NOs and the same two hard-rejection risks. Both flagged outcome maturity, interaction-vs-causation wording and compare alignment. Only the subagent raised oracle-data visibility; only Codex raised temporal customer identity.
- **VERDICT:** DESIGN CLEARED (PLAN; all 15 decisions auto-taken pending Adi's read). Eng review issues are mapped to tasks; eng review required.

NO UNRESOLVED DECISIONS
