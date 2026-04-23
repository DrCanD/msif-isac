"""Unit tests for H-LIF threshold functions."""

import numpy as np
import pytest

from msif_isac import thresholds as th


def test_det_above_threshold_fires():
    v = np.array([0.2, 0.14, 0.16])
    out = th.det(v, theta=0.15)
    assert out.tolist() == [True, False, True]


def test_det_has_no_randomness():
    v = np.array([0.16])
    a = th.det(v, 0.15)
    b = th.det(v, 0.15)
    assert (a == b).all()


def test_stoch_averages_toward_det_with_small_sigma():
    rng = np.random.default_rng(0)
    v = np.full(2000, 0.25, dtype=np.float32)
    # With sigma=0.01, firing rate at v=0.25 and theta=0.15 should be essentially 1.
    params = th.StochParams(sigma=0.01)
    fires = th.stoch(v, theta=0.15, params=params, rng=rng)
    assert fires.mean() > 0.99


def test_stoch_at_threshold_is_roughly_half():
    rng = np.random.default_rng(0)
    v = np.full(5000, 0.15, dtype=np.float32)
    params = th.StochParams(sigma=0.08)
    rate = th.stoch(v, theta=0.15, params=params, rng=rng).mean()
    assert 0.4 < rate < 0.6


def test_bh_fires_more_than_th_at_threshold():
    rng = np.random.default_rng(0)
    v = np.full(2000, 0.15, dtype=np.float32)
    # BH has aggressive hazard (lambda_max=3, delta=0.1), TH is gentle.
    r_bh = th.bh(v, 0.15, rng=rng).mean()
    r_th = th.th(v, 0.15, rng=rng).mean()
    assert r_bh > r_th


def test_registry_lookup():
    fn, cls = th.get_threshold("bh")
    assert fn is th.bh
    assert cls is th.BHParams


def test_unknown_threshold_raises():
    with pytest.raises(KeyError):
        th.get_threshold("zzz")


def test_is_stochastic_flags():
    assert not th.is_stochastic("det")
    assert th.is_stochastic("stoch")
    assert th.is_stochastic("bh")
    assert th.is_stochastic("th")
