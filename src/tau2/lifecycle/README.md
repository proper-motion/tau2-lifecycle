# tau2.lifecycle

Customer lifecycle simulator on top of τ²-bench telecom: persistent customers,
an authored churn hazard, CSAT, KPIs and matched campaigns across agent variants.
Design and decisions: `docs/lifecycle/PLAN.md` and
`docs/lifecycle/TOOLS_PLAN.md`.

## Secrets

API keys come from Doppler (project `gaia`, config `prd`, linked by `doppler.yaml`
at the repo root). Prefix every live command with `doppler run --`; it overrides
any key already exported in your shell.

```bash
doppler setup --no-interactive   # once per clone
```

## Run a campaign

```yaml
# cfg.yaml
name: smoke
customers: 1
days: 3
variants:
  - {name: mini, agent_llm: gpt-4.1-mini, failure_rate: 0.1}
user_llm: gpt-4.1-mini
csat_llm: gpt-4.1-mini
budget_usd: 0.5
task_split: small
```

```bash
doppler run -- uv run tau2 lifecycle run --config cfg.yaml --out runs/smoke
uv run tau2 lifecycle kpis --out runs/smoke --cutoff 3
uv run tau2 lifecycle cancel --out runs/smoke   # stops at the next day boundary
```

Running `run` again on the same `--out` resumes; a changed config is refused.
Measured cost (2026-10-27): about $0.024 per conversation with gpt-4.1-mini as
both agent and simulated customer.

## Tests

```bash
uv run pytest tests/lifecycle -q   # offline, no keys needed
```
