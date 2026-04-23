"""Sanity tests for the MS-IF neuron implementations."""

import numpy as np
import pytest

from msif_isac.neurons import baseline, dual_tau, chirp, phase, db, gabor
from msif_isac.neurons import get_neuron, REGISTRY


# ----------------------------------------------------------------------------
# Shape / type contracts
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    baseline.run, dual_tau.run, chirp.run, phase.run, db.run, gabor.run,
])
def test_output_shape_matches_input(fn):
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(1024).astype(np.float32)
    out = fn(sig, hlif="det")
    assert out.shape == sig.shape
    assert out.dtype == bool


@pytest.mark.parametrize("fn", [baseline.run, dual_tau.run, db.run, gabor.run])
def test_silent_input_produces_no_spikes(fn):
    sig = np.zeros(256, dtype=np.float32)
    out = fn(sig, hlif="det")
    assert not out.any()


def test_registry_covers_six_models():
    expected = {"baseline", "dual_tau", "chirp", "phase", "db", "gabor"}
    assert set(REGISTRY) == expected


def test_get_neuron_accepts_hyphenated_name():
    fn = get_neuron("dual-tau")
    assert fn is REGISTRY["dual_tau"]


def test_get_neuron_unknown_raises():
    with pytest.raises(KeyError):
        get_neuron("foobar")


# ----------------------------------------------------------------------------
# Functional sanity checks
# ----------------------------------------------------------------------------

def test_baseline_fires_on_large_input():
    sig = np.full(100, 0.3, dtype=np.float32)
    out = baseline.run(sig, theta=0.15, beta_m=0.9, refr=3, hlif="det")
    assert out.sum() > 0


def test_dual_tau_returns_two_channels_when_requested():
    sig = np.full(64, 0.5, dtype=np.float32)
    out = dual_tau.run(sig, hlif="det", return_channels=True)
    assert out.shape == (2, 64)


def test_chirp_resonates_at_tuned_frequency():
    # Input is a low-amplitude sinusoid at the tuned frequency: expect more
    # spikes than for a detuned input of the same amplitude. Amplitude is
    # kept small so the detuned case cannot saturate the refractory floor.
    n = 1024
    t = np.arange(n)
    f0 = 0.05
    amp = 0.05
    tuned = (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    detuned = (amp * np.sin(2 * np.pi * 0.4 * t)).astype(np.float32)

    s_tuned = chirp.run(tuned, f0_norm=f0, Q=30, theta=0.3, refr=5, hlif="det").sum()
    s_detuned = chirp.run(detuned, f0_norm=f0, Q=30, theta=0.3, refr=5, hlif="det").sum()
    assert s_tuned > s_detuned


def test_phase_responds_to_abrupt_phase_change():
    # Use a constant-phase analytic signal (f_ref cancels the carrier) so
    # the smooth case produces essentially no phase-deviation spikes.
    n = 512
    t = np.arange(n)
    f0 = 0.05
    smooth = np.exp(1j * 2 * np.pi * f0 * t)
    jumpy = smooth.copy()
    jumpy[n // 2:] *= np.exp(1j * np.pi)  # phase inversion mid-signal

    s_smooth = phase.run(
        smooth, f_ref_norm=f0, tau=8.0, dt_compare=1, phi_th=0.3, hlif="det"
    ).sum()
    s_jumpy = phase.run(
        jumpy, f_ref_norm=f0, tau=8.0, dt_compare=1, phi_th=0.3, hlif="det"
    ).sum()
    assert s_jumpy > s_smooth


def test_db_bistable_transitions():
    # Square-wave input should produce state transitions.
    sig = np.tile([0.0] * 20 + [0.3] * 20, 10).astype(np.float32)
    transitions = db.run(
        sig, beta_m=0.9, theta_high=0.15, theta_low=0.05, hlif="det"
    ).sum()
    assert transitions > 0


def test_gabor_envelope_fires():
    n = 512
    t = np.arange(n)
    burst = np.zeros(n, dtype=np.float32)
    burst[200:220] = np.sin(2 * np.pi * 0.05 * t[200:220])
    out = gabor.run(burst, sigma_t=8.0, f0_norm=0.05, theta=0.1, hlif="det")
    # Most spikes should cluster around the burst, not at the tails.
    assert out[100:300].sum() >= out[:100].sum() + out[300:].sum()
