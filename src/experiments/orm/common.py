"""Shared I/O and metrics for the ORM experiment (numpy only, no sklearn/scipy)."""

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

TARGETS = ("churn_or_downgrade_28d", "repeat_contact_7d", "csat")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC with tie handling. NaN if only one class present."""
    y, score = np.asarray(y, float), np.asarray(score, float)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = _ranks(score)
    return float(
        (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
    )


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(np.asarray(a, float)), _ranks(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def top_k_capture(true_harm: np.ndarray, score: np.ndarray, frac: float = 0.1) -> float:
    """Share of total true harm found in the top `frac` of rows ranked by `score`."""
    true_harm = np.asarray(true_harm, float)
    k = max(1, int(round(frac * len(true_harm))))
    picked = np.argsort(-np.asarray(score, float), kind="stable")[:k]
    total = true_harm.clip(min=0).sum()
    return (
        float(true_harm[picked].clip(min=0).sum() / total)
        if total > 0
        else float("nan")
    )


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):  # average ranks for ties
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def cluster_bootstrap(
    clusters: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and 95% CI, resampling whole clusters (customers, D17).

    `stat` receives row indices for one resample. Resamples where the statistic
    is undefined (NaN) are dropped.
    """
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(clusters)))
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        v = stat(np.concatenate([members[c] for c in pick]))
        if not np.isnan(v):
            boots.append(v)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(point), float(lo), float(hi)
