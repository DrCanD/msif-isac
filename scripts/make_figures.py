#!/usr/bin/env python
"""Reproduce the paper's figures from the JSON result files.

Usage:
    python scripts/make_figures.py --all --out figures/
    python scripts/make_figures.py --figure 3 --out figures/
    python scripts/make_figures.py --figure 4 --out figures/
    python scripts/make_figures.py --figure 5 --out figures/
    python scripts/make_figures.py --figure 6 --out figures/

Expects `results/{fmcw,dronerf,xiangyu}.json` from run_grand_cv.py,
`results/fft_baselines.json` from run_fft_baselines.py, and the audit
outputs `results/{dronerf_lofo,xiangyu_loso}.json` for Figure 4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from msif_isac.eval.grand_cv import MS_IF_MODELS, H_LIF_VARIANTS


ROW_LABELS = {"baseline": "Baseline", "phase": "Phase", "dual_tau": r"Dual-$\tau$",
              "chirp": "Chirp", "db": "DB", "gabor": "Gabor"}
COL_LABELS = {"det": "Det", "stoch": "Stoch", "bh": "BH", "th": "TH"}


def _matrix(records, vmax):
    """Return a 6x4 matrix of d-values and a mask for the winner cell."""
    M = np.full((6, 4), np.nan)
    for r in records:
        i = MS_IF_MODELS.index(r["ms_if"])
        j = H_LIF_VARIANTS.index(r["hlif"])
        M[i, j] = r["d"]
    win_idx = np.unravel_index(np.nanargmax(M), M.shape)
    return M, win_idx


def figure_3(results_dir: Path, out_dir: Path):
    """Grand cross-validation heatmaps (Figure 3)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    titles = ["FMCW Interference", "DroneRF Classification", "Xiangyu Automotive"]
    files = ["fmcw.json", "dronerf.json", "xiangyu.json"]
    vmaxes = [2.0, 0.7, 2.0]

    for ax, title, fname, vmax in zip(axes, titles, files, vmaxes):
        records = json.loads((results_dir / fname).read_text())
        M, win = _matrix(records, vmax)

        im = ax.imshow(M, aspect="auto", cmap="YlGn", vmin=0, vmax=vmax)
        ax.set_xticks(range(4))
        ax.set_xticklabels([COL_LABELS[h] for h in H_LIF_VARIANTS])
        ax.set_yticks(range(6))
        ax.set_yticklabels([ROW_LABELS[m] for m in MS_IF_MODELS])
        ax.set_title(title)
        ax.set_xlabel("H-LIF threshold")
        if ax is axes[0]:
            ax.set_ylabel("MS-IF feature")

        for i in range(6):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.plot(win[1], win[0], marker="*", markersize=16, markerfacecolor="gold",
                markeredgecolor="black", linestyle="none")
        fig.colorbar(im, ax=ax, label=r"$|d|$")

    fig.tight_layout()
    out = out_dir / "figure_3_grand_cv.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def figure_4(results_dir: Path, out_dir: Path):
    """LOFO / LOSO stability (Figure 4)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Xiangyu LOSO
    loso = json.loads((results_dir / "xiangyu_loso.json").read_text())
    # Pick the dual_tau_stoch record.
    rec = loso.get("dual_tau_stoch")
    if rec:
        seqs = [r["seq_id"] for r in rec["records"]]
        ds = [r["d"] for r in rec["records"]]
        cls_color = {"pedestrian": "#c66", "bicycle": "#6c6", "car": "#fc8"}
        colors = [cls_color.get(r["held_class"], "#999") for r in rec["records"]]
        axes[0].bar(range(len(seqs)), ds, color=colors, edgecolor="black")
        axes[0].axhline(rec["mean_d"], color="k", linestyle="-", linewidth=1,
                        label=f"mean {rec['mean_d']:.2f}")
        axes[0].axhline(0.8, color="red", linestyle=":", linewidth=1,
                        label="d = 0.8 threshold")
        axes[0].set_xticks(range(len(seqs)))
        axes[0].set_xticklabels(seqs, rotation=40, ha="right")
        axes[0].set_ylabel(r"$|d|$")
        axes[0].set_title(r"Xiangyu: Dual-$\tau$ $\times$ Stoch, 10-fold LOSO")
        axes[0].legend(fontsize=9)

    # (b) DroneRF top-5 LOFO
    lofo_data = json.loads((results_dir / "dronerf_lofo.json").read_text())
    top5 = sorted(lofo_data.values(), key=lambda r: -r["mean_d"])[:5]
    labels = [f"{r['ms_if']} × {r['hlif']}" for r in top5]
    means = [r["mean_d"] for r in top5]
    stds = [r["std_d"] for r in top5]
    axes[1].bar(range(len(labels)), means, yerr=stds, color="#8cf", edgecolor="black",
                capsize=4)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel(r"$|d|$  (mean ± std across LOFO folds)")
    axes[1].set_title("DroneRF: top-5 LOFO ranking")

    fig.tight_layout()
    out = out_dir / "figure_4_lofo_loso.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def figure_5(results_dir: Path, out_dir: Path):
    """MS-IF winners vs FFT baselines (Figure 5)."""
    fft = json.loads((results_dir / "fft_baselines.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    titles = ["FMCW Interference", "DroneRF Classification", "Xiangyu Automotive"]
    ds_names = ["fmcw", "dronerf", "xiangyu"]

    for ax, title, ds_name in zip(axes, titles, ds_names):
        records = json.loads((results_dir / f"{ds_name}.json").read_text())
        # Best MS-IF (excluding baseline)
        non_base = [r for r in records if r["ms_if"] != "baseline"]
        best = max(non_base, key=lambda r: r["d"])
        baseline_det = next(r for r in records
                            if r["ms_if"] == "baseline" and r["hlif"] == "det")

        bars = [
            ("Baseline × Det", baseline_det["d"], None, "gray"),
            (f"{best['ms_if']} × {best['hlif']}", best["d"],
             (best["ci_low"], best["ci_high"]), "#4cf"),
        ]
        fft_ds = fft.get(ds_name, {})
        for bl, col in [("energy", "#8f8"), ("fft_peak", "#fa8"),
                        ("fft_centroid", "#fa8"), ("fft_flatness", "#fa8"),
                        ("fft_bandwidth", "#fa8")]:
            if bl in fft_ds:
                bars.append((bl, fft_ds[bl]["d"],
                             (fft_ds[bl]["ci_low"], fft_ds[bl]["ci_high"]), col))

        labels = [b[0] for b in bars]
        vals = [b[1] for b in bars]
        errs_lo = [b[1] - b[2][0] if b[2] else 0 for b in bars]
        errs_hi = [b[2][1] - b[1] if b[2] else 0 for b in bars]
        colors = [b[3] for b in bars]
        ax.bar(range(len(bars)), vals, yerr=[errs_lo, errs_hi],
               color=colors, edgecolor="black", capsize=3)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        if ax is axes[0]:
            ax.set_ylabel(r"$|d|$")

    fig.tight_layout()
    out = out_dir / "figure_5_fft_vs_msif.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def figure_6(out_dir: Path):
    """Cost-accuracy landscape (Figure 6).

    This figure uses analytical operation counts (Table 7) rather than
    measured results. Numbers match the paper.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # (ops_per_chirp, d on FMCW, energy nJ, label, color)
    points = [
        (10240, 1.91, 20.7, "FFT (radix-2)",     "#fa8"),
        (1280,  1.49, 2.2,  "Baseline × Det",    "#aaa"),
        (2304,  1.96, 4.3,  "Dual-τ × Det",      "#4cf"),
        (1792,  1.95, 2.2,  "DB × Det",          "#4cf"),
        (7936,  0.73, 25.4, "Phase × Det",       "#4cf"),
        (22016, 1.56, 77.5, "Baseline × TH",     "#c8f"),
        (23040, 1.94, 79.6, "Dual-τ × TH",       "#c8f"),
        (40960, 1.34, 124.6, "Gabor × TH",       "#c8f"),
    ]

    for ops, d, nJ, label, color in points:
        ax.scatter(ops, d, s=np.sqrt(nJ) * 40, c=color, edgecolors="black",
                   zorder=3, alpha=0.9)
        ax.annotate(label, (ops, d), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)

    ax.axhline(1.91, color="orange", linestyle="--", linewidth=1, alpha=0.6,
               label="FFT accuracy")
    ax.axvspan(1e3, 5e3, ymin=0.85, alpha=0.08, color="green",
               label="Pareto-optimal region")
    ax.set_xscale("log")
    ax.set_xlim(8e2, 6e4)
    ax.set_ylim(0.5, 2.1)
    ax.set_xlabel("Operations per chirp (log scale)")
    ax.set_ylabel(r"Accuracy, $|d|$ on FMCW")
    ax.set_title("Cost-accuracy landscape (45 nm CMOS)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out = out_dir / "figure_6_cost_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figure", type=int, choices=[3, 4, 5, 6])
    p.add_argument("--all", action="store_true")
    p.add_argument("--results", default="results", help="Directory with JSON results.")
    p.add_argument("--out", default="figures", help="Output directory.")
    args = p.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [3, 4, 5, 6] if args.all else [args.figure]
    for fig_num in todo:
        if fig_num is None:
            continue
        try:
            if fig_num == 3: figure_3(results_dir, out_dir)
            elif fig_num == 4: figure_4(results_dir, out_dir)
            elif fig_num == 5: figure_5(results_dir, out_dir)
            elif fig_num == 6: figure_6(out_dir)
        except FileNotFoundError as e:
            print(f"[Figure {fig_num}] skipped: {e}")


if __name__ == "__main__":
    main()
