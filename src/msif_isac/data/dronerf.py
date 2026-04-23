"""DroneRF dataset (Al-Sa'd et al., Mendeley Data 10.17632/f4c2b4n755.1).

Four classes: AR, Bebop, Phantom, Background. Five files per class in
the canonical release. The paper uses 50 windows per class across 5 files
(primary protocol) plus a leave-one-file-out audit over 20 splits
(4 classes x 5 files).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


CLASSES = ("AR", "Bebop", "Phantom", "Background")
WINDOW_LEN = 50_000  # samples per window (matches paper)


def load_windows(
    data_dir: str | Path,
    n_windows_per_file: int = 10,
    stride_fraction: float = 0.0,
) -> dict[str, list[np.ndarray]]:
    """Return per-class, per-file lists of windowed segments.

    Returns
    -------
    dict[class_label, list of 2-D arrays]
        Each list entry corresponds to one file; each 2-D array is
        (n_windows, WINDOW_LEN).

    Notes
    -----
    Window stride was tested at 0% and 50% overlap in the paper; scores did
    not decrease with overlap, ruling out window-overlap leakage.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"DroneRF data not found at {data_dir}. "
            f"See docs/datasets.md for download instructions."
        )

    out: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {cls_dir} to contain the class recordings. "
                f"Adapt load_windows() to your local layout if needed."
            )
        files = sorted(cls_dir.glob("*.npy"))  # or .csv / .mat as needed
        if not files:
            raise FileNotFoundError(f"No recordings found in {cls_dir}")
        for f in files:
            sig = np.load(f).astype(np.float32)
            step = max(1, int(WINDOW_LEN * (1.0 - stride_fraction)))
            wins = []
            for start in range(0, len(sig) - WINDOW_LEN + 1, step):
                wins.append(sig[start:start + WINDOW_LEN])
                if len(wins) >= n_windows_per_file:
                    break
            if wins:
                out[cls].append(np.stack(wins))
    return out


def flatten(windows: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Collapse per-file lists into one array per class (for primary protocol)."""
    return {cls: np.concatenate(lst, axis=0) for cls, lst in windows.items() if lst}


def lofo_splits(windows: dict[str, list[np.ndarray]]):
    """Yield (train_dict, test_dict) pairs for leave-one-file-out.

    For each (class, file_index) pair, build a test set containing only
    that one file (its windows) while every other file from that class
    and all files from other classes go into the training pool.
    """
    for cls, files in windows.items():
        for i, test_arr in enumerate(files):
            test = {cls: test_arr}
            train = {}
            for other_cls, other_files in windows.items():
                remainder = [a for j, a in enumerate(other_files)
                             if not (other_cls == cls and j == i)]
                if remainder:
                    train[other_cls] = np.concatenate(remainder, axis=0)
            yield train, test, (cls, i)
