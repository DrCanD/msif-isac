# Datasets

The paper uses three publicly available datasets (60 GB total). None of the raw data is redistributed here. Download each one from its host and stage it under `./data/` as described below. All three loaders can be pointed to a different root via the `dataset.data_dir` key in the YAML configs.

## 1. FMCW radar interference mitigation (Rock et al., 2020)

IEEE DataPort, DOI [10.21227/1fhk-b416](https://doi.org/10.21227/1fhk-b416). 2.37 GB.

Expected layout:

```
data/fmcw/
├── frames.npy         # (n_frames, 1024) float32 magnitude frames
└── README.md          # optional metadata
```

The `src/msif_isac/data/fmcw.py` loader currently expects a single `frames.npy` file. If your copy uses a different on-disk format (MAT, HDF5), replace the body of `load_frames()` with the appropriate reader. The return contract is fixed: a `(n_frames, frame_len)` float32 array.

Clean vs. interference labels are produced unsupervisedly by `split_clean_interference()` using a 3× median-power threshold (Section 6.8). A threshold sweep at 1.5×, 2×, 3×, 4×, 5× gives the stable plateau reported in the paper. The `flatness` pseudo-label is available as a robustness check.

## 2. DroneRF (Al-Sa'd et al., 2019)

Mendeley Data, DOI [10.17632/f4c2b4n755.1](https://doi.org/10.17632/f4c2b4n755.1). 43 GB.

Expected layout:

```
data/dronerf/
├── AR/
│   ├── 10000L_0.npy
│   ├── 10000L_1.npy
│   └── ...
├── Bebop/
├── Phantom/
└── Background/
```

Each class directory contains the raw RF recordings. Five files per class is the canonical release. The loader reads each file, segments it into 50,000-sample windows (0% or 50% stride), and returns a per-class, per-file dictionary. LOFO splits iterate over `(class, file_index)` pairs.

## 3. Xiangyu raw ADC 77 GHz FMCW (Gao et al., 2022)

IEEE DataPort, DOI [10.21227/xm40-jx59](https://doi.org/10.21227/xm40-jx59). ~15 GB.

Expected layout:

```
data/xiangyu/
├── pms1000.npy
├── pms2000.npy
├── pms3000.npy
├── bms1000.npy
├── bm1s007.npy
├── cms1000.npy
├── css1000.npy
├── cm1s000.npy
├── cm1s003.npy
└── cm1s014.npy
```

Each `.npy` holds either a 3-D tensor shaped `(n_frames, 255, 128)` or a 2-D tensor `(n_frames, 32640)`. DC subtraction is applied per chirp at load time. The sequence IDs above are the 10 pure single-class sequences used for LOSO (3 pedestrian, 2 bicycle, 5 car).

## Excluded pilot datasets

Four datasets were piloted during scope-boundary exploration and excluded from the main evaluation. Section 6.8 documents the reasons; scripts for the exclusion analyses are not included in this repo but are available on request.

- 5G NR IQ (phase-invariance rescoping)
- RadChar synthetic pulsed radar (FFT-optimal by construction)
- DeepSense 6G Scenarios 42–44 (mmWave ISAC beyond single-neuron scope)
- Xiangyu pbms002 binary (single-pedestrian, not multi-class)

## Adapting to other datasets

To add a new dataset:

1. Write a loader in `src/msif_isac/data/` returning either `(a, b)` for a two-class protocol or `{class_label: array_(n_samples, frame_len)}` for multi-class.
2. Add a YAML config under `configs/` with per-dataset membrane parameters and `neuron_kwargs`.
3. Extend `scripts/run_grand_cv.py::load_dataset()` to dispatch on the new name.

The evaluation pipeline (`msif_isac.eval.grand_cv`) is dataset-agnostic once the loader returns the expected shape.
