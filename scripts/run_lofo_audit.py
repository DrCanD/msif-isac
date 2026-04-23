#!/usr/bin/env python
"""Leave-one-file-out (DroneRF) and leave-one-sequence-out (Xiangyu) audits.

Usage:
    python scripts/run_lofo_audit.py --dataset dronerf --out results/dronerf_lofo.json
    python scripts/run_lofo_audit.py --dataset xiangyu --loso --out results/xiangyu_loso.json

The audits reproduce Figure 4 of the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from msif_isac.data import dronerf, xiangyu
from msif_isac.eval import lofo, loso
from msif_isac.eval.grand_cv import MS_IF_MODELS, H_LIF_VARIANTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["dronerf", "xiangyu"], required=True)
    p.add_argument("--loso", action="store_true",
                   help="Leave-one-sequence-out (Xiangyu only).")
    p.add_argument("--out", required=True)
    p.add_argument("--top_k", type=int, default=5,
                   help="Evaluate only the top-k cells (by primary d) for speed.")
    p.add_argument("--primary_results", default=None,
                   help="JSON from run_grand_cv.py. If missing, evaluate all 24 cells.")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(f"configs/{args.dataset}.yaml").read_text())
    kwargs_by_model = cfg["neuron_kwargs"]

    # Pick which (ms_if, hlif) cells to audit.
    if args.primary_results is not None:
        primary = json.loads(Path(args.primary_results).read_text())
        primary_sorted = sorted(primary, key=lambda r: -r["d"])
        cells = [(r["ms_if"], r["hlif"]) for r in primary_sorted[:args.top_k]]
    else:
        cells = [(m, h) for m in MS_IF_MODELS for h in H_LIF_VARIANTS]

    audit_results = {}

    if args.dataset == "dronerf":
        windows = dronerf.load_windows(
            cfg["dataset"]["data_dir"],
            n_windows_per_file=cfg["dataset"].get("n_windows_per_file", 10),
            stride_fraction=cfg["dataset"].get("stride_fraction", 0.0),
        )
        for ms_if, hlif in cells:
            kw = kwargs_by_model.get(ms_if, {})
            mean_d, std_d, per_fold = lofo.lofo_audit(windows, ms_if, hlif, kw)
            audit_results[f"{ms_if}_{hlif}"] = {
                "ms_if": ms_if, "hlif": hlif,
                "mean_d": mean_d, "std_d": std_d, "per_fold": per_fold,
            }
            print(f"  {ms_if:>9s} x {hlif:>5s}:  "
                  f"mean d = {mean_d:5.3f} ± {std_d:5.3f}  "
                  f"(n_folds = {len(per_fold)})")

    elif args.dataset == "xiangyu":
        if not args.loso:
            raise SystemExit("Xiangyu audit requires --loso (leave-one-sequence-out).")
        per_seq = xiangyu.load_all(cfg["dataset"]["data_dir"])
        for ms_if, hlif in cells:
            kw = kwargs_by_model.get(ms_if, {})
            mean_d, std_d, records = loso.loso_audit(per_seq, ms_if, hlif, kw)
            audit_results[f"{ms_if}_{hlif}"] = {
                "ms_if": ms_if, "hlif": hlif,
                "mean_d": mean_d, "std_d": std_d, "records": records,
            }
            print(f"  {ms_if:>9s} x {hlif:>5s}:  "
                  f"mean d = {mean_d:5.3f} ± {std_d:5.3f}  "
                  f"(n_folds = {len(records)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit_results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
