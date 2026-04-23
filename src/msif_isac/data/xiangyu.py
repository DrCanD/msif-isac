"""Xiangyu raw ADC 77 GHz FMCW (Gao et al., IEEE DataPort 10.21227/xm40-jx59).

Three classes: pedestrian, bicycle, car. The paper uses 10 pure single-class
sequences (3 pedestrian, 2 bicycle, 5 car) for leave-one-sequence-out
cross-validation. Each frame combines 255 chirps x 128 ADC samples = 32640
samples, DC-subtracted per chirp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


CHIRPS_PER_FRAME = 255
SAMPLES_PER_CHIRP = 128
FRAME_LEN = CHIRPS_PER_FRAME * SAMPLES_PER_CHIRP  # 32640

SEQUENCES = {
    "pedestrian": ["pms1000", "pms2000", "pms3000"],
    "bicycle":    ["bms1000", "bm1s007"],
    "car":        ["cms1000", "css1000", "cm1s000", "cm1s003", "cm1s014"],
}


def load_sequence(data_dir: str | Path, seq_id: str) -> np.ndarray:
    """Load a single recording as a (n_frames, FRAME_LEN) float32 array.

    DC subtraction is applied per chirp (mean across the 128 ADC samples).
    """
    data_dir = Path(data_dir)
    path = data_dir / f"{seq_id}.npy"  # replace with the actual on-disk format
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. See docs/datasets.md for how to stage the "
            f"Xiangyu raw-ADC release into `{data_dir}/`."
        )

    raw = np.load(path).astype(np.float32)
    # Expect shape (n_frames, CHIRPS_PER_FRAME, SAMPLES_PER_CHIRP) or
    # (n_frames, FRAME_LEN). Handle both.
    if raw.ndim == 3:
        raw = raw - raw.mean(axis=-1, keepdims=True)
        raw = raw.reshape(raw.shape[0], FRAME_LEN)
    elif raw.ndim == 2 and raw.shape[1] == FRAME_LEN:
        # Already flattened. Apply DC subtraction per chirp.
        per_chirp = raw.reshape(-1, CHIRPS_PER_FRAME, SAMPLES_PER_CHIRP)
        per_chirp = per_chirp - per_chirp.mean(axis=-1, keepdims=True)
        raw = per_chirp.reshape(-1, FRAME_LEN)
    else:
        raise ValueError(
            f"Unexpected shape {raw.shape} for {seq_id}. Expected "
            f"(n_frames, {CHIRPS_PER_FRAME}, {SAMPLES_PER_CHIRP}) or "
            f"(n_frames, {FRAME_LEN})."
        )
    return raw


def load_all(data_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Return {class_label: {seq_id: frames_array}} for all sequences."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for cls, seq_ids in SEQUENCES.items():
        out[cls] = {sid: load_sequence(data_dir, sid) for sid in seq_ids}
    return out


def flatten(per_seq: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Pool all sequences per class (primary protocol)."""
    return {cls: np.concatenate(list(d.values()), axis=0) for cls, d in per_seq.items()}


def loso_splits(per_seq: dict[str, dict[str, np.ndarray]]):
    """Yield (train_dict, test_dict, held_out_id) for leave-one-sequence-out.

    Each iteration holds out one single-class sequence while pooling the rest.
    """
    for cls, seqs in per_seq.items():
        for seq_id, frames in seqs.items():
            test = {cls: frames}
            train = {}
            for other_cls, other_seqs in per_seq.items():
                remainder = [arr for sid, arr in other_seqs.items()
                             if not (other_cls == cls and sid == seq_id)]
                if remainder:
                    train[other_cls] = np.concatenate(remainder, axis=0)
            yield train, test, seq_id
