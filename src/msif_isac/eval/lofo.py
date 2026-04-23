"""Leave-one-file-out audit for DroneRF.

For each (class, file) pair, the target file is held out and used as the
test fold. The remaining files across all classes form the training pool.
The paper reports 20 splits total (4 classes x 5 files) and a top-4 cluster
within d in [0.66, 0.70] for the winning cells.
"""

from __future__ import annotations

import numpy as np

from msif_isac import metrics
from msif_isac.data import dronerf
from msif_isac.eval.grand_cv import spike_feature


def lofo_audit(
    windows: dict[str, list[np.ndarray]],
    ms_if: str,
    hlif: str,
    neuron_kwargs: dict,
    seed: int = 42,
) -> tuple[float, float, list[float]]:
    """Return (mean_d, std_d, per_fold_d) across LOFO splits.

    The metric is mean pairwise |Cohen's d| on the held-out fold, where the
    held-out class provides one array and every other class is pooled
    from the training partition. This is a conservative reading: we measure
    how well the cell separates the held-out file from the pooled non-class.
    """
    per_fold = []
    rng = np.random.default_rng(seed)
    for train, test, _tag in dronerf.lofo_splits(windows):
        # Features on training partition (pooled per class).
        train_feats = {
            cls: spike_feature(arr, ms_if, neuron_kwargs, hlif, int(rng.integers(1, 10**6)))
            for cls, arr in train.items()
        }
        # Features on held-out fold.
        (held_cls, held_arr), = test.items()
        held_feat = spike_feature(
            held_arr, ms_if, neuron_kwargs, hlif, int(rng.integers(1, 10**6))
        )
        # Mean pairwise d: held-out class vs. each other training class.
        ds = [
            metrics.cohens_d(held_feat, train_feats[c])
            for c in train_feats if c != held_cls
        ]
        per_fold.append(float(np.mean(ds)) if ds else 0.0)

    arr = np.array(per_fold)
    return float(arr.mean()), float(arr.std(ddof=1)), per_fold
