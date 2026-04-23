#!/usr/bin/env python
"""Run the 24-cell grand cross-validation for one dataset.

Usage:
    python scripts/run_grand_cv.py --config configs/fmcw.yaml --out results/fmcw.json

The three per-dataset runs together reproduce the 72-cell matrix in
Figure 3 and Table 5 of the paper.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from msif_isac.eval import grand_cv
from msif_isac.data import fmcw, dronerf, xiangyu


def load_dataset(cfg: dict):
    name = cfg["dataset"]["name"]
    ddir = cfg["dataset"]["data_dir"]

    if name == "fmcw":
        frames = fmcw.load_frames(ddir, cfg["dataset"].get("frame_len", 1024))
        clean, inter = fmcw.split_clean_interference(
            frames,
            pseudo=cfg["dataset"].get("pseudo_label", "power"),
            threshold_mult=cfg["dataset"].get("threshold_mult", 3.0),
        )
        return "two_class", (clean, inter)

    if name == "dronerf":
        windows = dronerf.load_windows(
            ddir,
            n_windows_per_file=cfg["dataset"].get("n_windows_per_file", 10),
            stride_fraction=cfg["dataset"].get("stride_fraction", 0.0),
        )
        return "multi_class", dronerf.flatten(windows)

    if name == "xiangyu":
        per_seq = xiangyu.load_all(ddir)
        return "multi_class", xiangyu.flatten(per_seq)

    raise ValueError(f"Unknown dataset: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    kind, data = load_dataset(cfg)

    def progress(r):
        print(
            f"  {r.ms_if:>9s} x {r.hlif:>5s}:  "
            f"d = {r.d:6.3f}  CI [{r.ci_low:6.3f}, {r.ci_high:6.3f}]  "
            f"(n_runs={r.n_runs})"
        )

    print(f"[{cfg['dataset']['name']}] 24-cell grand CV")
    if kind == "two_class":
        class_a, class_b = data
        print(f"  two-class: {len(class_a)} vs {len(class_b)} frames")
        results = grand_cv.grand_cv_two_class(
            class_a, class_b,
            cfg["neuron_kwargs"],
            n_runs=cfg["eval"].get("n_runs", 3),
            n_boot=cfg["eval"].get("n_boot", 5000),
            progress=progress,
        )
    else:
        print("  multi-class:")
        for cls, arr in data.items():
            print(f"    {cls:>12s}: {len(arr)} samples")
        results = grand_cv.grand_cv_multi_class(
            data,
            cfg["neuron_kwargs"],
            n_runs=cfg["eval"].get("n_runs", 3),
            n_boot=cfg["eval"].get("n_boot", 5000),
            progress=progress,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
