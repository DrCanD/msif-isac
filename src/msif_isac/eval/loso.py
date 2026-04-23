"""Leave-one-sequence-out audit for Xiangyu.

Each of the 10 single-class sequences is held out in turn; the remainder
is pooled per class. The paper reports Dual-tau x Stoch mean d = 1.80 +/- 0.17
across the 10 folds, all above the d = 0.8 sanity threshold.
"""

from __future__ import annotations

import numpy as np

from msif_isac import metrics
from msif_isac.data import xiangyu
from msif_isac.eval.grand_cv import spike_feature


def loso_audit(
    per_seq: dict[str, dict[str, np.ndarray]],
    ms_if: str,
    hlif: str,
    neuron_kwargs: dict,
    seed: int = 42,
) -> tuple[float, float, list[dict]]:
    """Return (mean_d, std_d, per_fold_records)."""
    records = []
    per_fold_d = []
    rng = np.random.default_rng(seed)
    for train, test, seq_id in xiangyu.loso_splits(per_seq):
        (held_cls, held_arr), = test.items()
        train_feats = {
            cls: spike_feature(arr, ms_if, neuron_kwargs, hlif, int(rng.integers(1, 10**6)))
            for cls, arr in train.items()
        }
        held_feat = spike_feature(
            held_arr, ms_if, neuron_kwargs, hlif, int(rng.integers(1, 10**6))
        )
        ds = [
            metrics.cohens_d(held_feat, train_feats[c])
            for c in train_feats if c != held_cls
        ]
        d_fold = float(np.mean(ds)) if ds else 0.0
        per_fold_d.append(d_fold)
        records.append({"seq_id": seq_id, "held_class": held_cls, "d": d_fold})

    arr = np.array(per_fold_d)
    return float(arr.mean()), float(arr.std(ddof=1)), records
