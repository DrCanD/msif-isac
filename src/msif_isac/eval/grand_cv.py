"""Grand cross-validation: 6 MS-IF x 4 H-LIF = 24 cells per dataset.

Applied to three datasets, yielding the 72-cell matrix reported in
Figure 3 and Table 5 of the paper.

Stochastic models (Stoch, BH, TH) are 3-run averaged. Bootstrap 95% CIs
use 5000 iterations for FMCW/DroneRF and 1000 for Xiangyu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from msif_isac import metrics, thresholds
from msif_isac.neurons import get_neuron


MS_IF_MODELS = ("baseline", "dual_tau", "chirp", "phase", "db", "gabor")
H_LIF_VARIANTS = ("det", "stoch", "bh", "th")


@dataclass
class CellResult:
    ms_if: str
    hlif: str
    d: float
    ci_low: float
    ci_high: float
    n_runs: int


def spike_feature(
    signal_batch: np.ndarray,
    ms_if: str,
    neuron_kwargs: dict,
    hlif: str,
    seed: int,
) -> np.ndarray:
    """Run an MS-IF neuron on each signal in a batch and return spike counts.

    Spike count is the discrimination feature used throughout the paper.
    """
    fn = get_neuron(ms_if)
    rng = np.random.default_rng(seed)
    counts = np.empty(len(signal_batch), dtype=np.float64)
    for i, sig in enumerate(signal_batch):
        spikes = fn(sig, hlif=hlif, rng=rng, **neuron_kwargs)
        counts[i] = int(np.sum(spikes))
    return counts


def evaluate_cell_two_class(
    class_a: np.ndarray,
    class_b: np.ndarray,
    ms_if: str,
    hlif: str,
    neuron_kwargs: dict,
    n_runs: int = 3,
    n_boot: int = 5000,
) -> CellResult:
    """Two-class Cohen's d (FMCW interference protocol)."""
    seeds = [42 + k for k in range(n_runs)] if thresholds.is_stochastic(hlif) else [42]

    def one_run(seed: int) -> float:
        fa = spike_feature(class_a, ms_if, neuron_kwargs, hlif, seed)
        fb = spike_feature(class_b, ms_if, neuron_kwargs, hlif, seed)
        return metrics.cohens_d(fa, fb)

    ds = np.array([one_run(s) for s in seeds])
    mean_d = float(ds.mean())

    # Bootstrap CI with the first seed's feature vectors (paper protocol).
    fa0 = spike_feature(class_a, ms_if, neuron_kwargs, hlif, seeds[0])
    fb0 = spike_feature(class_b, ms_if, neuron_kwargs, hlif, seeds[0])
    lo, hi = metrics.bootstrap_ci(metrics.cohens_d, fa0, fb0, n_iter=n_boot)

    return CellResult(ms_if, hlif, mean_d, lo, hi, len(seeds))


def evaluate_cell_multi_class(
    classes: dict[str, np.ndarray],
    ms_if: str,
    hlif: str,
    neuron_kwargs: dict,
    n_runs: int = 3,
    n_boot: int = 5000,
) -> CellResult:
    """Mean pairwise Cohen's d (DroneRF, Xiangyu protocol)."""
    seeds = [42 + k for k in range(n_runs)] if thresholds.is_stochastic(hlif) else [42]

    def compute(seed: int) -> tuple[float, dict[str, np.ndarray]]:
        feats = {
            cls: spike_feature(arr, ms_if, neuron_kwargs, hlif, seed)
            for cls, arr in classes.items()
        }
        return metrics.mean_pairwise_d(feats), feats

    ds = []
    features = None
    for s in seeds:
        d, feats = compute(s)
        ds.append(d)
        if features is None:
            features = feats

    mean_d = float(np.mean(ds))

    # Bootstrap across pairs using the first seed's features.
    boot = np.empty(n_boot, dtype=np.float64)
    rng = np.random.default_rng(42)
    arrays = list(features.values())
    for i in range(n_boot):
        resampled = {
            cls: rng.choice(arr, size=len(arr), replace=True)
            for cls, arr in zip(features, arrays)
        }
        boot[i] = metrics.mean_pairwise_d(resampled)
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))

    return CellResult(ms_if, hlif, mean_d, lo, hi, len(seeds))


def grand_cv_two_class(
    class_a: np.ndarray,
    class_b: np.ndarray,
    neuron_kwargs_by_model: dict[str, dict],
    n_runs: int = 3,
    n_boot: int = 5000,
    progress: Callable | None = None,
) -> list[CellResult]:
    """Run the 24-cell matrix for a two-class dataset (FMCW)."""
    out: list[CellResult] = []
    for ms_if in MS_IF_MODELS:
        kw = neuron_kwargs_by_model.get(ms_if, {})
        for hlif in H_LIF_VARIANTS:
            r = evaluate_cell_two_class(
                class_a, class_b, ms_if, hlif, kw, n_runs, n_boot
            )
            out.append(r)
            if progress is not None:
                progress(r)
    return out


def grand_cv_multi_class(
    classes: dict[str, np.ndarray],
    neuron_kwargs_by_model: dict[str, dict],
    n_runs: int = 3,
    n_boot: int = 5000,
    progress: Callable | None = None,
) -> list[CellResult]:
    """Run the 24-cell matrix for a multi-class dataset (DroneRF, Xiangyu)."""
    out: list[CellResult] = []
    for ms_if in MS_IF_MODELS:
        kw = neuron_kwargs_by_model.get(ms_if, {})
        for hlif in H_LIF_VARIANTS:
            r = evaluate_cell_multi_class(
                classes, ms_if, hlif, kw, n_runs, n_boot
            )
            out.append(r)
            if progress is not None:
                progress(r)
    return out
