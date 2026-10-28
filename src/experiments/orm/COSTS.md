# Cost guide: lifecycle campaigns for the ORM experiment

Measured on pilot 1 (2026-10-28): 30 customers × 9 agent variants × 60 virtual
days, 716 conversations, $96.19 total, 0 infra errors. Models: agents gpt-4.1 /
gpt-4.1-mini / gpt-5, simulated customer gpt-4.1, CSAT rater gpt-4.1-mini.

## Unit costs

| Unit | Cost |
|---|---|
| One conversation, average | $0.134 |
| …with gpt-5 as agent | $0.187 |
| …with gpt-4.1 as agent | $0.139 |
| …with gpt-4.1-mini as agent | $0.085 |
| One churn-or-downgrade event (any split) | ~13 conversations ≈ **$1.78** |
| One churn event in the **test split** (15% of customers) | ≈ **$12** |

A "churn event" is one customer copy (customer × variant) that cancels or
downgrades. Pilot rate: 54 of 270 copies in 60 days (1.8 per population customer).

## Size → cost (60-day horizon, same models)

This table counts all churn events. The primary verdict now scores support calls
only (deviation 5); use `RUNBOOK.md`'s size table, which is in those terms.

| Test-split events | Customers | Conversations | API cost | Smallest ORM-vs-judge gap it can reliably show* |
|---|---|---|---|---|
| 50 | ~185 | ~4.4k | ~$600 | ~0.12 AUROC |
| 100 | ~370 | ~8.9k | ~$1,200 | ~0.09 AUROC |
| 200 (preregistered) | ~740 | ~17.7k | ~$2,400 | ~0.06 AUROC |
| 400 | ~1,480 | ~35k | ~$4,700 | ~0.04 AUROC |

*Rough rule of thumb for a paired AUROC difference with customer-level
bootstrap; treat as order-of-magnitude, not a power analysis.

Add about 15-20% for the held-out simulation (D16) if it is run, and about
$1-3 per 1,000 rows for each LLM-judge arm. ORM training is separate GPU time
(roughly $10-20 per run on one H100).

## Levers

- **Cheaper agents** cut cost most: gpt-5 calls are 2.2× gpt-4.1-mini calls.
- **Fewer variants** (for example drop the 10% failure rate) cut cost by a third
  with little loss of quality spread.
- **Longer horizon** (90 days instead of 60) gives more churn events per
  customer, so fewer customers are needed for the same event count.
- **Simulated-customer model**: gpt-4.1-mini instead of gpt-4.1 roughly halves
  the customer side of every call, but departs from τ²-bench's default.

## Wall-clock

About 150-170 conversations per hour at `max_concurrency: 8`, rising roughly
linearly with concurrency until provider rate limits. The Mac must stay awake:
lid-close sleep pauses the run; resume continues from the last completed day.
