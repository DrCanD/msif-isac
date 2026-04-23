"""Discrimination metrics used in the paper.

- cohens_d: standardized two-sample effect size (absolute).
- mean_pairwise_d: mean |Cohen's d| across all class pairs (DroneRF, Xiangyu).
- bootstrap_ci: bootstrap 95% confidence interval on a statistic.
- permutation_p: p-value under class-label shuffling.
- multi_run_average: 3-run average wrapper for stochastic thresholds.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from itertools import combinations


# ----------------------------------------------------------------------------
# Effect sizes
# ----------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Cohen's d between two samples (pooled SD)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp < 1e-12:
        return 0.0
    return float(abs(a.mean() - b.mean()) / sp)


def mean_pairwise_d(classes: dict[str, np.ndarray]) -> float:
    """Mean |Cohen's d| across all unordered class pairs.

    Parameters
    ----------
    classes : mapping label -> 1-D array of per-sample feature values.
    """
    labels = list(classes)
    vals = [cohens_d(classes[a], classes[b]) for a, b in combinations(labels, 2)]
    return float(np.mean(vals)) if vals else 0.0


# ----------------------------------------------------------------------------
# Bootstrap and permutation
# ----------------------------------------------------------------------------

def bootstrap_ci(
    statistic_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_iter: int = 5000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap percentile CI for a two-or-more-sample statistic.

    Each array is resampled with replacement independently at each iteration.
    """
    rng = rng or np.random.default_rng(42)
    arrays = [np.asarray(a) for a in arrays]
    boot = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        resampled = [rng.choice(a, size=len(a), replace=True) for a in arrays]
        boot[i] = statistic_fn(*resampled)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def permutation_p(
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    a: np.ndarray,
    b: np.ndarray,
    n_iter: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, np.ndarray]:
    """Two-sample permutation test. Returns (p_value, null_distribution).

    The observed statistic is compared against the distribution obtained by
    shuffling class labels. Used in the paper for Xiangyu (p < 5e-4 reported
    across the four Dual-τ variants).
    """
    rng = rng or np.random.default_rng(42)
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    observed = statistic_fn(a, b)
    combined = np.concatenate([a, b])
    n_a = len(a)
    null = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        rng.shuffle(combined)
        null[i] = statistic_fn(combined[:n_a], combined[n_a:])
    # One-sided p for |effect| kind of statistics (Cohen's d is |·|).
    p = float((null >= observed).mean())
    # Avoid reporting exactly 0 for finite permutation count.
    p = max(p, 1.0 / n_iter)
    return p, null


# ----------------------------------------------------------------------------
# Stochastic-model averaging
# ----------------------------------------------------------------------------

def multi_run_average(
    fn: Callable[[int], float],
    n_runs: int = 3,
    seeds: Iterable[int] | None = None,
) -> tuple[float, float]:
    """Average a stochastic evaluation over n_runs with distinct seeds.

    Returns
    -------
    (mean, std) of the runs.
    """
    seeds = list(seeds) if seeds is not None else list(range(n_runs))
    runs = np.array([fn(s) for s in seeds], dtype=np.float64)
    return float(runs.mean()), float(runs.std(ddof=1)) if n_runs > 1 else 0.0
