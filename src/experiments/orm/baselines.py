"""Reference predictors: random, observable-feature regression, task-success-only.

The observable baseline sees only what the judges see (the transcript) and uses
counts anyone could compute. The task-success-only baseline reads the private
native reward, so it is a scoring reference, never a fair competitor.
Every predictor writes rows of {row_id, p_churn, p_repeat, csat_pred}.
"""

import re

import numpy as np

FEATURES = (
    "turns",
    "tool_calls",
    "tool_errors",
    "transfer",
    "cancel_words",
    "length_k",
)


def observable_features(transcript: str) -> np.ndarray:
    lines = transcript.splitlines()
    return np.array(
        [
            sum(1 for x in lines if x.startswith(("assistant:", "user:"))),
            sum(1 for x in lines if " -> " in x),
            sum(1 for x in lines if x.startswith("tool:") and "Error" in x),
            int("transfer_to_human_agents" in transcript),
            len(
                re.findall(
                    r"\b(cancel|close my|switch provider|leave)\b", transcript, re.I
                )
            ),
            len(transcript) / 1000,
        ],
        float,
    )


class Logistic:
    """L2-regularised logistic regression by gradient descent on standardised X."""

    def __init__(self, l2: float = 1.0, steps: int = 2000, lr: float = 0.1):
        self.l2, self.steps, self.lr = l2, steps, lr

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logistic":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        w = np.zeros(Z.shape[1])
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-Z @ w))
            grad = Z.T @ (p - y) / len(y) + self.l2 * np.r_[0, w[1:]] / len(y)
            w -= self.lr * grad
        self.w = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        return 1 / (1 + np.exp(-Z @ self.w))


def observable_predictions(train: list[dict], test: list[dict]) -> list[dict]:
    Xtr = np.array([observable_features(r["transcript"]) for r in train])
    Xte = np.array([observable_features(r["transcript"]) for r in test])
    out = {r["row_id"]: {"row_id": r["row_id"]} for r in test}
    for target, key in (
        ("churn_or_downgrade_28d", "p_churn"),
        ("repeat_contact_7d", "p_repeat"),
    ):
        y = np.array([r[target] for r in train], float)
        p = (
            Logistic().fit(Xtr, y).predict_proba(Xte)
            if 0 < y.mean() < 1
            else np.full(len(test), y.mean())
        )
        for r, v in zip(test, p):
            out[r["row_id"]][key] = float(v)
    rated = [(x, r["csat"]) for x, r in zip(Xtr, train) if r["csat"] is not None]
    if rated:
        A = np.c_[np.ones(len(rated)), np.array([x for x, _ in rated])]
        coef, *_ = np.linalg.lstsq(
            A, np.array([c for _, c in rated], float), rcond=None
        )
        csat = np.clip(np.c_[np.ones(len(Xte)), Xte] @ coef, 1, 5)
    else:
        csat = np.full(len(test), 3.0)
    for r, v in zip(test, csat):
        out[r["row_id"]]["csat_pred"] = float(v)
    return list(out.values())


def random_predictions(test: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        {
            "row_id": r["row_id"],
            "p_churn": float(rng.random()),
            "p_repeat": float(rng.random()),
            "csat_pred": float(rng.uniform(1, 5)),
        }
        for r in test
    ]


def task_success_predictions(test: list[dict], private: dict[str, dict]) -> list[dict]:
    """Reference only: reads the private native reward."""
    out = []
    for r in test:
        s = private[r["row_id"]]["task_success"]
        s = 0.5 if s is None else s
        out.append(
            {
                "row_id": r["row_id"],
                "p_churn": 1 - s,
                "p_repeat": 1 - s,
                "csat_pred": 1 + 4 * s,
            }
        )
    return out
