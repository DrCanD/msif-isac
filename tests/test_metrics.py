"""Tests for the discrimination metrics."""

import numpy as np
import pytest

from msif_isac import metrics


def test_cohens_d_zero_for_identical_samples():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.cohens_d(a, a) == 0.0


def test_cohens_d_symmetry():
    a = np.random.default_rng(0).standard_normal(50)
    b = np.random.default_rng(1).standard_normal(50) + 1.0
    assert metrics.cohens_d(a, b) == pytest.approx(metrics.cohens_d(b, a))


def test_cohens_d_scales_with_mean_gap():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(200)
    b_small = rng.standard_normal(200) + 0.5
    b_large = rng.standard_normal(200) + 2.0
    assert metrics.cohens_d(a, b_large) > metrics.cohens_d(a, b_small)


def test_mean_pairwise_d_on_three_classes():
    rng = np.random.default_rng(0)
    classes = {
        "A": rng.standard_normal(80),
        "B": rng.standard_normal(80) + 1.5,
        "C": rng.standard_normal(80) + 3.0,
    }
    d = metrics.mean_pairwise_d(classes)
    assert d > 1.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(200)
    b = rng.standard_normal(200) + 1.0
    d_hat = metrics.cohens_d(a, b)
    lo, hi = metrics.bootstrap_ci(metrics.cohens_d, a, b, n_iter=500)
    assert lo <= d_hat <= hi


def test_permutation_p_is_small_for_clear_separation():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(100)
    b = rng.standard_normal(100) + 3.0
    p, null = metrics.permutation_p(metrics.cohens_d, a, b, n_iter=500)
    assert p < 0.05
    assert len(null) == 500


def test_permutation_p_is_not_tiny_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(100)
    b = rng.standard_normal(100)
    p, _ = metrics.permutation_p(metrics.cohens_d, a, b, n_iter=500)
    assert p > 0.1


def test_multi_run_average_reports_std():
    fn = lambda s: np.random.default_rng(s).standard_normal()
    mean, std = metrics.multi_run_average(fn, n_runs=5)
    assert isinstance(mean, float)
    assert std >= 0.0
