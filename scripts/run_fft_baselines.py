#!/usr/bin/env python
"""Compute FFT-based and non-FFT baselines on all three datasets (Table 6).

Usage:
    python scripts/run_fft_baselines.py --all --out results/fft_baselines.json
    python scripts/run_fft_baselines.py --dataset fmcw --out results/fmcw_fft.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from msif_isac import metrics
from msif_isac.baselines import fft_features
from msif_isac.data import fmcw, dronerf, xiangyu


BASELINES = ["energy", "fft_energy", "fft_peak", "fft_centroid", "fft_flatness", "fft_bandwidth"]


def score_two_class(a: np.ndarray, b: np.ndarray, baseline: str, n_boot: int = 5000):
    fn = fft_features.get_baseline(baseline)
    fa = np.array([fn(x) for x in a])
    fb = np.array([fn(x) for x in b])
    d = metrics.cohens_d(fa, fb)
    lo, hi = metrics.bootstrap_ci(metrics.cohens_d, fa, fb, n_iter=n_boot)
    return d, lo, hi


def score_multi_class(classes: dict[str, np.ndarray], baseline: str, n_boot: int = 5000):
    fn = fft_features.get_baseline(baseline)
    feats = {cls: np.array([fn(x) for x in arr]) for cls, arr in classes.items()}
    d = metrics.mean_pairwise_d(feats)

    # Bootstrap over pairs.
    rng = np.random.default_rng(42)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        resampled = {cls: rng.choice(arr, size=len(arr), replace=True)
                     for cls, arr in feats.items()}
        boot[i] = metrics.mean_pairwise_d(resampled)
    return d, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def evaluate_dataset(name: str, cfg_path: str | Path, n_boot: int):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    ddir = cfg["dataset"]["data_dir"]

    print(f"[{name}] FFT baselines")
    if name == "fmcw":
        frames = fmcw.load_frames(ddir)
        clean, inter = fmcw.split_clean_interference(
            frames,
            pseudo=cfg["dataset"].get("pseudo_label", "power"),
            threshold_mult=cfg["dataset"].get("threshold_mult", 3.0),
        )
        scorer = lambda bl: score_two_class(clean, inter, bl, n_boot)
    elif name == "dronerf":
        windows = dronerf.load_windows(
            ddir,
            n_windows_per_file=cfg["dataset"].get("n_windows_per_file", 10),
            stride_fraction=cfg["dataset"].get("stride_fraction", 0.0),
        )
        data = dronerf.flatten(windows)
        scorer = lambda bl: score_multi_class(data, bl, n_boot)
    elif name == "xiangyu":
        per_seq = xiangyu.load_all(ddir)
        data = xiangyu.flatten(per_seq)
        scorer = lambda bl: score_multi_class(data, bl, n_boot)
    else:
        raise ValueError(name)

    out = {}
    for bl in BASELINES:
        d, lo, hi = scorer(bl)
        print(f"  {bl:>14s}: d = {d:6.3f}  CI [{lo:6.3f}, {hi:6.3f}]")
        out[bl] = {"d": d, "ci_low": lo, "ci_high": hi}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fmcw", "dronerf", "xiangyu"])
    p.add_argument("--all", action="store_true", help="Run all three datasets.")
    p.add_argument("--out", required=True)
    p.add_argument("--n_boot", type=int, default=5000)
    args = p.parse_args()

    results = {}
    datasets = ["fmcw", "dronerf", "xiangyu"] if args.all else [args.dataset]
    for name in datasets:
        n_boot = 1000 if name == "xiangyu" else args.n_boot
        results[name] = evaluate_dataset(name, f"configs/{name}.yaml", n_boot)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
