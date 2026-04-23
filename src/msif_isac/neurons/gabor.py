"""Gabor-LIF. Complex Gabor filter feeding a LIF.

Paper Section 4.5, Eqs. (8)-(9):

    g(t) = exp(-t^2 / (2 * sigma_t^2)) * exp(j * 2*pi * f_0 * t)
    e(t) = |I(t) * g(t)|,   tau dv/dt = -v + e(t),   spike if v >= theta

Biological mapping: mammalian auditory cortex spectro-temporal receptive
fields (STRFs), well approximated by Gabor functions (deCharms et al. 1998,
Shamma 2001, Theunissen et al. 2001). The Gabor kernel achieves the
theoretical lower bound of the Heisenberg uncertainty principle for joint
time-frequency localization.
"""

from __future__ import annotations

import numpy as np

from msif_isac import thresholds as th
from msif_isac.neurons.baseline import run as baseline_run


def gabor_kernel(sigma_t: float, f0_norm: float, truncate: float = 4.0) -> np.ndarray:
    """Build a complex Gabor kernel centred on t=0, sampled per-timestep."""
    half = int(np.ceil(truncate * sigma_t))
    t = np.arange(-half, half + 1, dtype=np.float64)
    envelope = np.exp(-(t ** 2) / (2.0 * sigma_t ** 2))
    carrier = np.exp(1j * 2.0 * np.pi * f0_norm * t)
    kern = envelope * carrier
    # Unit-energy normalization so the envelope e(t) has a stable magnitude scale.
    kern /= np.linalg.norm(kern) + 1e-12
    return kern.astype(np.complex64)


def run(
    signal: np.ndarray,
    sigma_t: float = 8.0,
    f0_norm: float = 0.05,
    beta_m: float = 0.9,
    theta: float = 0.15,
    refr: int = 3,
    hlif: str = "det",
    hlif_params=None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run a Gabor-LIF on the input signal.

    The Gabor envelope e(t) = |I(t) * g(t)| is fed into the standard LIF.
    """
    signal = np.asarray(signal)
    if not np.iscomplexobj(signal):
        signal = signal.astype(np.complex64)

    kern = gabor_kernel(sigma_t, f0_norm)
    # 'same' convolution preserves length.
    conv = np.convolve(signal, kern, mode="same")
    envelope = np.abs(conv).astype(np.float32)

    return baseline_run(
        envelope,
        beta_m=beta_m,
        theta=theta,
        refr=refr,
        hlif=hlif,
        hlif_params=hlif_params,
        rng=rng,
    )
