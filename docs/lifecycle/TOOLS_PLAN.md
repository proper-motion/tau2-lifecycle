> Design plan and review record for the lifecycle simulator and ORM experiment.
> Copied from the author's working notes on 2026-10-28; local paths and private repo names removed.

# Plan: telecom lifecycle tools (plan change, cancel, billing cycle)

Repo: proper-motion/tau2-lifecycle (fork of sierra-research/tau2-bench at ebf990b). Parent plan: `.plans/tau-lifecycle-orm.md`.
Status: REVIEWED 2026-10-27. L1-L4 built (commit on proper-motion/tau2-lifecycle main, 26 lifecycle tests + 231 upstream focused tests green). L5 done in 47effbb: the 4 OV7 end-to-end tests pass. Complete.

## Problem

The lifecycle sim decides downgrade, cancel and renew, but τ²-bench telecom cannot express them:
- Agent tools (13 `@is_tool` methods in `src/tau2/domains/telecom/tools.py`): no change-plan, no cancel, no renew, no goodwill credit.
- Policy (`data/tau2/domains/telecom/main_policy.md:141`) has a "Change Plan" section the agent cannot execute; no cancel section.
- Tasks: 2,285, all tech support (mms 1,984, mobile data 254, service 47). None about plans or cancelling.
- Data model already has what's needed: `LineStatus.CLOSED`, `AccountStatus.CLOSED`, `Line.contract_end_date`, `Line.last_plan_change_date`, `Customer.goodwill_credit_used_this_year`, 5 plans.
- No billing cycle: `Line.data_used_gb` never resets, so a downgrade to Basic (5 GB) with 8.7 GB used leaves only 23/114 templates applicable (Phase 1 finding).

Today a downgrade or cancel would be a silent ledger write. That means the agent never handles the moments that matter most for churn, and the ORM never sees those traces.

## Proposed change (draft)

1. **Agent tools** (WRITE), added to telecom:
   - `change_plan(customer_id, line_id, new_plan_id)`: line must be Active, plan must exist and differ. Sets `plan_id`, `last_plan_change_date`. Returns old/new monthly price.
   - `cancel_line(customer_id, line_id, reason)`: line must not already be Closed. Sets status Closed. If it was the customer's last open line, sets the account to Closed.
   - `apply_goodwill_credit(customer_id, amount)`: retention lever. Capped per year, tracked in `goodwill_credit_used_this_year`.
   - `renew_contract(customer_id, line_id, months)`: extends `contract_end_date`.
2. **Policy**: add "Cancel Line" (confirm intent, offer one retention credit within cap, then cancel if the customer still wants to) and "Goodwill Credit" sections. Keep "Change Plan" as-is.
3. **Lifecycle-intent tasks**: a generator for conversations the customer starts *because of* a lifecycle decision: "downgrade to Basic", "upgrade", "cancel my line", "renew". Each has persona variants and DB-state grading (expected plan/status after the call).
4. **Billing cycle** (lifecycle layer, not a tool): at each virtual month boundary and on plan change, reset `data_used_gb` and `data_refueling_gb` to 0.
5. **Lifecycle integration**: when the hazard model says "downgrade" or "cancel", schedule an intent conversation instead of writing the state directly. The resulting DB state (plan changed? line closed? saved by a credit?) becomes the lifecycle outcome.
6. **Tests**: tool unit tests (success and each failure check), intent task generation, gold-path applicability for the new tasks, billing reset.

## Open questions for review

- Edit upstream `telecom` in place, or add a separate domain?
- Should every lifecycle action go through a conversation, or only some?
- Which of the four tools are needed now?

## Review decisions (2026-10-27)

Adi: no renew tool; then "I don't care about how you do it as long as it is correct". The remaining choices took the recommended option and are marked (auto).

| ID | Decision | How chosen |
|---|---|---|
| D1 | **No `renew_contract` tool.** Renewal is the default: a line stays open unless cancelled. | Adi |
| D2 | **Include `apply_goodwill_credit`** as the one retention lever, so cancel calls test agent skill. | auto |
| D3 | **Separate domain `telecom_lifecycle`, upstream `telecom` untouched.** One new module, `src/tau2/lifecycle/telecom_ext.py`, holds a `TelecomTools` subclass with the new tools, a policy addendum appended to upstream policy text, an environment constructor and the intent-task generator. It is registered in `registry.py` (one line). This keeps the D13 contract: upstream telecom's tool list, policy and task scores are unchanged. | auto |
| D4 | **Downgrade and cancel go through a conversation.** When the hazard model picks downgrade or cancel, the customer calls with that intent, and the DB state after the call is the outcome: plan changed, line closed, or saved by a credit. "Stopped using the product" stays silent (no call). | auto |
| D5 | **Retention calls grade deterministically.** The policy fixes one offer (a $20 credit, $50/year cap including already-used credit). The customer's instructions say accept or refuse, from their hidden price sensitivity (accept if above 0.5). Gold path: accept → `apply_goodwill_credit(20)`, line stays Active; refuse → `cancel_line`. | auto |
| D6 | **Billing cycle in the lifecycle layer.** Every 30 virtual days, reset `data_used_gb` and `data_refueling_gb` to 0. **Not on plan change** (revised by OV5: resetting on change would hand out free data). After a mid-cycle downgrade the customer is over the cap until the cycle ends, as with a real carrier. Evidence: on Basic, applicable templates are 23/114 before the reset and 58/114 after (probe 2026-10-27). | auto, revised |

### Findings

**Architecture**
1. [P1] (confidence 9/10) Editing upstream `tools.py`/`main_policy.md` would add tools and policy text to every upstream telecom task. That breaks the approved D13 contract ("the only intended upstream change is registering the lifecycle CLI subcommand") and changes native task-success comparability. Resolved by D3.
2. [P1] (8/10) Silent downgrade/cancel writes hide the moments where agent behavior most affects churn from the trace and from the ORM. Resolved by D4.
3. [P1] (8/10) Retention calls need a deterministic gold path, or native grading is undefined. Resolved by D5.
4. [P2] (9/10) No billing cycle: downgraded customers lose most task types. Resolved by D6. Remaining limit: Basic still excludes 56/114 templates whose init sets usage between 5 and 15 GB; acceptable.

**Code quality**
1. [P2] (8/10) The new tools follow the existing pattern (`@is_tool(ToolType.WRITE)`, `_get_target_line`, `ValueError` on failed checks, `regional_settings.format_money`). No new helpers.
2. [P3] (7/10) Phase 1's `check_applicable` overlaps upstream `TaskManager.verify_task`/`run_assertions` (`tasks/manager.py:170-215`). Upstream's version asserts and prints, so it isn't reusable as-is. Shared-code extraction rejected; keep ours.

**Tests** (required proof of the decisions above)

```
telecom_ext.py
  change_plan        [GAP] success; same plan; unknown plan; line not Active
  cancel_line        [GAP] success; already Closed; last open line closes the account
  apply_goodwill     [GAP] success; over annual cap (incl. pre-used 25.0); negative amount
  policy addendum    [GAP] present in telecom_lifecycle, absent in telecom
  intent generator   [GAP] deterministic; persona variants; accept and refuse variants
  gold applicability [GAP] every generated intent task passes check_applicable
world.py
  billing reset      [GAP] usage zeroed at day 30k and on plan change
regression (D13)   [GAP] upstream telecom tool list = 13 tools, policy text unchanged,
                          retarget equivalence still green
```

**Performance:** no issues (all in-memory, per conversation).

### Outside voice (Codex gpt-6-astra, completed: 1 Critical, 6 High, 2 Medium)

All resolved below under Adi's "as long as it's correct" instruction (auto, recommended).

| OV | Finding | Resolution |
|---|---|---|
| 1 Critical | An open line after a cancel call could mean "saved", or tool failure, timeout or obstruction, so obstruction would look like retention | **Outcome = the customer's decision, recorded explicitly.** `cancelled`: line closed by the agent. `retained`: customer instructed to accept AND credit applied in that call. `cancel_blocked`: line still open with no valid save (agent refused or errored, call timed out). `cancel_blocked` counts as churn: the line closes the next day via self-serve, CSAT is forced to 1, and the call's labels are negative. Obstruction can never become a positive label. |
| 2 High | What a hazard draw means after an intervention | **The hazard draw is the intent to cancel.** An accepted offer suppresses cancel intent for 30 virtual days; after that the hazard runs as normal. The suppression is logged in the oracle. |
| 3 High | Impossible gold paths (cap exhausted, suspended line) | **Eligibility is computed from the customer's actual state** when the task is built. The offer is made only if the line is Active and the remaining annual cap is ≥ $20. Otherwise the policy says no offer and the gold path is a plain cancel. Accept/refuse applies only when eligible. |
| 4 High | Credit accounting undefined | **The credit is a −$20 line item on the customer's next bill.** `goodwill_credit_used_this_year` resets every 365 virtual days. The counter is per customer; cancellation is per line. |
| 5 High | Resetting usage on plan change gives free data | D6 revised: reset only at the 30-day cycle. |
| 6 High | A fixed threshold tests script-following, not retention skill | Accepted as a limit. Native grading of the call only checks policy compliance. The ORM's signal is later outcomes (D7 windows). PREREGISTRATION must not claim "retention skill". |
| 7 High | Integration deferred; tool tests can pass with no lifecycle effect | **L1-L4 are not done until L5's end-to-end tests pass**: downgrade, accepted retention, refused retention, failed call. Each runs against projected customer state, persists the mutation to the ledger and checks the checkpoint label. |
| 8 Medium | Cancellation's downstream meaning undefined | **Closing L1002 (the customer's line) = customer churn:** billing stops and no future contacts. Closing other lines doesn't count. "Stopped using" means usage drops while the customer is still billed and still contactable. |
| 9 Medium | D3 doesn't settle D13 | **D13 contract amended:** allowed upstream changes are the CLI subcommand plus one `registry.py` line for `telecom_lifecycle`. The regression test builds both envs: `telecom` has 13 tools and an unchanged policy, retarget equivalence stays green, and `telecom_lifecycle` has 16 tools. |

### Implementation tasks

- [ ] **L1 (P1, human ~1d / CC ~45min)** telecom_ext.py: `change_plan`, `cancel_line`, `apply_goodwill_credit` + tool tests
- [ ] **L2 (P1, human ~4h / CC ~20min)** Policy addendum (Cancel Line, Goodwill Credit) + `telecom_lifecycle` env + registry line + D13 regression test
- [ ] **L3 (P1, human ~1d / CC ~1h)** Intent task generator (downgrade, upgrade, cancel × persona × accept/refuse) + gold applicability test
- [ ] **L4 (P1, human ~2h / CC ~15min)** world.py billing reset + test
- [ ] **L5 (Phase 2, gates L1-L4 done)** checkpoint.py: hazard downgrade/cancel → intent conversation (D4); outcome categories per OV1; 30-day suppression per OV2; credit as next-bill line item (OV4); 4 end-to-end tests (OV7)

### NOT in scope

- `renew_contract` (D1)
- Upgrade/downgrade initiated by the agent as a sales action
- Retention offers other than the single fixed credit

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Outside Review | Codex via `/plan-eng-review` outside voice | Independent 2nd opinion | 1 | completed (issues_found) | 9 findings (1 Critical, 6 High, 2 Medium), all resolved |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES OPEN (mapped to L1-L5) | 15 issues (4 arch, 2 code quality, 8 test gaps, 1 integration), 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — (no UI) | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **OUTSIDE COVERAGE:** codex (openai, gpt-6-astra), plan-review phase, completed, 9 findings.
- **CROSS-MODEL:** Both reviews found the D13 isolation and gold-path issues. Codex alone caught label semantics (OV1, OV2), credit accounting, the free-data loophole and the deferred end-to-end proof.
- **VERDICT:** No review is CLEAR. Findings are mapped to tasks L1-L5; eng review required.

NO UNRESOLVED DECISIONS
