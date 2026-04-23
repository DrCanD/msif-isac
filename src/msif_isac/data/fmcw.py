"""FMCW radar interference dataset (Rock et al., IEEE DataPort 10.21227/1fhk-b416).

Two classes: clean vs. interference-affected frames. The paper uses an
unsupervised 3x median-power pseudo-label (Section 6.8). A sensitivity
sweep is documented at 1.5x, 2x, 3x, 4x, 5x thresholds, with the 3x choice
on a stable plateau.

The loader returns per-frame magnitude samples. Downstream MS-IF models
and FFT baselines operate on these frames directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


PseudoLabel = Literal["power", "flatness"]


def load_frames(
    data_dir: str | Path,
    frame_len: int = 1024,
    max_frames: int | None = None,
) -> np.ndarray:
    """Load FMCW chirp frames from the dataset directory.

    The specific on-disk format depends on how the IEEE DataPort release
    is extracted. This stub expects per-frame files or a single .npy/.mat
    archive at `data_dir`. Replace the body with the user's actual
    preprocessing code. The return contract is fixed: float32 array
    shaped (n_frames, frame_len).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"FMCW data not found at {data_dir}. "
            f"See docs/datasets.md for download instructions."
        )

    # Placeholder implementation. Replace with the actual loader for your
    # local copy of the dataset.
    # Example for a .npy archive containing a (n_frames, frame_len) matrix:
    candidate = data_dir / "frames.npy"
    if candidate.exists():
        frames = np.load(candidate).astype(np.float32)
        if max_frames is not None:
            frames = frames[:max_frames]
        return frames

    raise FileNotFoundError(
        f"Expected {candidate} with a (n_frames, {frame_len}) array. "
        f"Adapt load_frames() in {__file__} to match your local layout."
    )


def split_clean_interference(
    frames: np.ndarray,
    pseudo: PseudoLabel = "power",
    threshold_mult: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split frames into (clean, interference) by an unsupervised pseudo-label.

    Parameters
    ----------
    frames : (n_frames, frame_len) array.
    pseudo : 'power' (default, matches paper) or 'flatness' (robustness check).
    threshold_mult : multiple of median used as the cut. 3.0 is the paper
        default; 2.0-5.0 yields the stable plateau documented in Section 6.8.

    Returns
    -------
    clean, interference : two 2-D arrays with the classified frames.
    """
    if pseudo == "power":
        scores = np.mean(frames ** 2, axis=1)
    elif pseudo == "flatness":
        # log flatness per frame; extreme values flag anomalous frames.
        p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        p = np.maximum(p, 1e-20)
        log_gm = np.mean(np.log(p), axis=1)
        am = p.mean(axis=1)
        scores = -np.exp(log_gm) / (am + 1e-30)  # negate to keep "high = anomalous"
    else:
        raise ValueError(f"Unknown pseudo-label '{pseudo}'")

    cut = threshold_mult * np.median(scores)
    interference_mask = scores >= cut
    return frames[~interference_mask], frames[interference_mask]
