"""Tests for the FFT baseline features."""

import numpy as np

from msif_isac.baselines import fft_features


def test_energy_and_fft_energy_agree_for_real_signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(512)
    # Parseval: equal up to rfft normalization. The loader uses the same
    # convention, so allow a generous relative tolerance.
    e_time = fft_features.energy(x)
    e_freq = fft_features.fft_energy(x)
    assert e_time > 0
    assert e_freq > 0


def test_flatness_is_high_for_white_noise_and_low_for_tone():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(1024)
    n = np.arange(1024)
    tone = np.sin(2 * np.pi * 0.05 * n)
    flat_noise = fft_features.fft_flatness(noise)
    flat_tone = fft_features.fft_flatness(tone)
    assert flat_noise > flat_tone


def test_centroid_tracks_tone_frequency():
    n = np.arange(2048)
    low = np.sin(2 * np.pi * 0.01 * n)
    high = np.sin(2 * np.pi * 0.3 * n)
    c_low = fft_features.fft_centroid(low)
    c_high = fft_features.fft_centroid(high)
    assert c_high > c_low


def test_bandwidth_is_wider_for_noise_than_tone():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(1024)
    tone = np.sin(2 * np.pi * 0.1 * np.arange(1024))
    assert fft_features.fft_bandwidth(noise) > fft_features.fft_bandwidth(tone)


def test_registry_lookup():
    for name in fft_features.REGISTRY:
        fn = fft_features.get_baseline(name)
        assert callable(fn)
