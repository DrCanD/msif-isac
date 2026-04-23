#!/usr/bin/env python
"""27-configuration grid search for membrane parameters (Section 5.1).

Validates that the chosen per-dataset parameters lie on a plateau rather
than a singular optimum. The paper reports the FMCW Dual-τ x Det grid
optimum at d = 1.970 vs. d = 1.957 at the chosen configuration — a 0.7%
gap within bootstrap noise.

Usage:
    python scripts/run_hyperparameter_sweep.py --dataset fmcw --cell dual_tau_det
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from msif_isac import metrics
from msif_isac.data import fmcw, dronerf, xiangyu
from msif_isac.eval.grand_cv import spike_feature


BETA_GRID = [0.85, 0.9, 0.95, 0.98]
THETA_GRID = [0.05, 0.1, 0.15, 0.2]
REFR_GRID = [1, 2, 3, 5]


def load_dataset(name: str, cfg: dict):
    ddir = cfg["dataset"]["data_dir"]
    if name == "fmcw":
        frames = fmcw.load_frames(ddir)
        clean, inter = fmcw.split_clean_interference(
            frames,
            pseudo=cfg["dataset"].get("pseudo_label", "power"),
            threshold_mult=cfg["dataset"].get("threshold_mult", 3.0),
        )
        return "two_class", (clean, inter)
    if name == "dronerf":
        w = dronerf.load_windows(ddir, cfg["dataset"].get("n_windows_per_file", 10))
        return "multi_class", dronerf.flatten(w)
    if name == "xiangyu":
        return "multi_class", xiangyu.flatten(xiangyu.load_all(ddir))
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fmcw", "dronerf", "xiangyu"], required=True)
    p.add_argument("--cell", required=True,
                   help="MS-IF x H-LIF cell, e.g. 'dual_tau_det'.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ms_if, hlif = args.cell.rsplit("_", 1)
    cfg = yaml.safe_load(Path(f"configs/{args.dataset}.yaml").read_text())
    kind, data = load_dataset(args.dataset, cfg)

    print(f"[{args.dataset}] grid sweep for {ms_if} x {hlif}")
    print(f"  grid = {len(BETA_GRID) * len(THETA_GRID) * len(REFR_GRID)} configs")

    results = []
    for beta_m, theta, refr in itertools.product(BETA_GRID, THETA_GRID, REFR_GRID):
        # Override the membrane params but keep MS-IF-specific feature params.
        kw = dict(cfg["neuron_kwargs"].get(ms_if, {}))
        kw.update({"beta_m": beta_m, "theta": theta, "refr": refr})
        # Phase and DB use their own theta-like names; remap transparently.
        if ms_if == "phase":
            kw["phi_th"] = theta
        if ms_if == "db":
            kw["theta_high"] = theta
            kw["theta_low"] = max(theta / 3, 0.02)

        if kind == "two_class":
            a, b = data
            fa = spike_feature(a, ms_if, kw, hlif, seed=42)
            fb = spike_feature(b, ms_if, kw, hlif, seed=42)
            d = metrics.cohens_d(fa, fb)
        else:
            feats = {cls: spike_feature(arr, ms_if, kw, hlif, seed=42)
                     for cls, arr in data.items()}
            d = metrics.mean_pairwise_d(feats)

        results.append({"beta_m": beta_m, "theta": theta, "refr": refr, "d": d})
        print(f"  beta_m={beta_m:4.2f}  theta={theta:4.2f}  refr={refr}  ->  d={d:6.3f}")

    results_sorted = sorted(results, key=lambda r: -r["d"])
    top = results_sorted[0]
    print(f"\nGrid optimum: d={top['d']:.3f} at beta_m={top['beta_m']}, "
          f"theta={top['theta']}, refr={top['refr']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results_sorted, indent=2))
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
