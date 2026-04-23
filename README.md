# msif-isac

Reproduction code for the paper:

> Dikmen, İ. C. (2026). **Does Nature Know Fourier? Lessons from Echolocating Species for FFT-Free Neuromorphic Radar and ISAC.** *(Manuscript under review.)*

This repository reproduces the 72-cell grand cross-validation (6 MS-IF × 4 H-LIF × 3 datasets), the FFT baseline comparisons, the leave-one-file-out and leave-one-sequence-out audits, and all figures in the paper.

The MS-IF and H-LIF neuron models used here are a subset of the broader [`dikmen-spiking-neurons`](https://github.com/DrCanD/dikmen-spiking-neurons) library (39 models across the H-LIF, MS-IF, and related families). This repo carries only the models and thresholds that appear in the paper; the full library carries the rest.

## What is in the paper

Two orthogonal design axes for neuromorphic radar neurons:

- **MS-IF (Multi-Scale Integrate-and-Fire)** — what temporal features the neuron listens to. Five specialized models plus a baseline LIF: Dual-τ, Chirp, Phase, DB, Gabor, Baseline.
- **H-LIF (Hazard-Based LIF)** — how the neuron decides to fire. Four threshold variants: Det, Stoch, BH, TH.

Three real-world datasets (60 GB total):

- FMCW radar interference mitigation (Rock et al., IEEE DataPort 10.21227/1fhk-b416)
- DroneRF (Al-Sa'd et al., Mendeley Data 10.17632/f4c2b4n755.1)
- Xiangyu raw ADC 77 GHz FMCW (Gao et al., IEEE DataPort 10.21227/xm40-jx59)

## Installation

Python 3.10+ required. Clone the repo and install:

```bash
git clone https://github.com/candikmen/msif-isac.git
cd msif-isac
pip install -e .
```

To run tests:

```bash
pip install -e ".[dev]"
pytest tests/
```

## Data preparation

The three datasets are not redistributed here. See `docs/datasets.md` for download instructions and expected directory layout. Default paths are `./data/fmcw/`, `./data/dronerf/`, `./data/xiangyu/`. Override via the config files in `configs/`.

## Reproducing the paper

The four scripts below reproduce the results tables and figures. Stochastic models (Stoch, BH, TH) are 3-run averaged by default. Bootstrap resampling uses 5000 iterations for FMCW/DroneRF and 1000 for Xiangyu.

```bash
# 72-cell grand cross-validation (Table 5, Figure 3)
python scripts/run_grand_cv.py --config configs/fmcw.yaml --out results/fmcw.json
python scripts/run_grand_cv.py --config configs/dronerf.yaml --out results/dronerf.json
python scripts/run_grand_cv.py --config configs/xiangyu.yaml --out results/xiangyu.json

# FFT baselines (Table 6, Figure 5)
python scripts/run_fft_baselines.py --all --out results/fft_baselines.json

# LOFO / LOSO audits (Figure 4)
python scripts/run_lofo_audit.py --dataset dronerf --out results/dronerf_lofo.json
python scripts/run_lofo_audit.py --dataset xiangyu --loso --out results/xiangyu_loso.json

# Hyperparameter sweep validation (Section 5.1 footnote)
python scripts/run_hyperparameter_sweep.py --dataset fmcw --cell dual_tau_det

# Figures
python scripts/make_figures.py --all
```

Total wall-clock on a single modern CPU: roughly 4–6 hours end to end. No GPU required.

## Default parameters

H-LIF parameters are fixed across datasets (Section 5.1):

| Variant | Parameters |
|---------|-----------|
| Det     | (none) |
| Stoch   | σ = 0.08 |
| BH      | λmax = 3, κ = 5, η = 3, δ = 0.1 |
| TH      | λmax = 0.5, κ = 1, η = 3, δ = 0.02 |

Membrane parameters are per dataset, set by a 27-configuration grid search on a calibration subset:

| Dataset  | βm   | θ    | refr |
|----------|------|------|------|
| FMCW     | 0.9  | 0.15 | 3    |
| DroneRF  | 0.9  | 0.05 | 3    |
| Xiangyu  | 0.85 | 0.15 | 2    |

MS-IF feature parameters (ftune, τ, barrier thresholds) are set by biological analogy or signal physics and are not optimized.

## Citation

If this code helps your work, please cite the paper:

```bibtex
@article{dikmen2026nature,
  author  = {Dikmen, İsmail Can},
  title   = {Does Nature Know Fourier? Lessons from Echolocating Species for FFT-Free Neuromorphic Radar and ISAC},
  journal = {(Manuscript under review)},
  year    = {2026}
}
```

and, if you use neurons from the broader library:

```bibtex
@software{dikmen2026snlib,
  author  = {Dikmen, İsmail Can},
  title   = {dikmen-spiking-neurons: A library of 39 spiking neuron models for neuromorphic signal processing},
  year    = {2026},
  url     = {https://github.com/DrCanD/dikmen-spiking-neurons},
  license = {Apache-2.0}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

The Apache 2.0 license includes an explicit patent grant to users of this software. A patent application covering the TH-LIF neuron model and the MS-IF neuron family is in preparation with the Turkish Patent and Trademark Office (TÜRKPATENT). The license permits open-source use and redistribution under the terms of Apache 2.0.

## Author

**İsmail Can Dikmen** — Assistant Professor, Electrical and Electronics Engineering, İstinye University, Istanbul, Turkey.

- ORCID: [0000-0002-7747-7777](https://orcid.org/0000-0002-7747-7777)
- Web: [Google Scholar](https://scholar.google.com/citations?user=c4OrnOQAAAAJ&lan=en)
- IEEE Senior Member
