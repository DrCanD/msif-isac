"""Chirp-LIF. Resonant beat-frequency tuned neuron.

Paper Section 4.2, Eq. (3):

    dz/dt = (-1/tau + j*omega_0) z + I(t),   spike if |z| >= theta

Complex-valued membrane z accumulates energy at the tuned frequency
omega_0 = 2*pi*f_0. Quality factor Q = omega_0 * tau / 2.

Biological mapping: horseshoe bat DSCF neurons with Q_{10dB} up to 500.
For FMCW radar, the beat frequency f_b = 2*R*B/(c*T) is proportional
to target range R; a bank of Chirp-LIF neurons tuned to different f_0
produces a spiking range profile without FFT.
"""

from __future__ import annotations

import numpy as np

from msif_isac import thresholds as th


def run(
    signal: np.ndarray,
    f0_norm: float = 0.05,
    Q: float = 30.0,
    theta: float = 0.15,
    refr: int = 3,
    hlif: str = "det",
    hlif_params=None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run a Chirp-LIF (resonate-and-fire) on the input signal.

    Parameters
    ----------
    signal : 1-D array (real or complex).
    f0_norm : tuned frequency, normalized to the sampling rate (cycles/sample).
    Q      : quality factor.
    theta  : firing threshold on |z|.
    refr   : refractory period in timesteps.
    hlif, hlif_params, rng : as elsewhere.

    Returns
    -------
    spikes : boolean array, same length as signal.
    """
    signal = np.asarray(signal)
    if not np.iscomplexobj(signal):
        signal = signal.astype(np.complex64)

    omega = 2.0 * np.pi * f0_norm
    tau = 2.0 * Q / omega
    # Discrete update: z_{t+1} = z_t * (1 - 1/tau + j*omega) + I
    # For numerical stability, use the matrix exponential of the continuous op.
    a = np.exp((-1.0 / tau + 1j * omega) * 1.0)  # dt = 1 sample

    fire_fn, param_cls = th.get_threshold(hlif)
    if hlif_params is None and hlif != "det":
        hlif_params = param_cls()

    z = 0.0 + 0.0j
    cooldown = 0
    spikes = np.zeros(signal.shape[0], dtype=bool)

    for t, x in enumerate(signal):
        if cooldown > 0:
            cooldown -= 1
            z *= a
            continue

        z = a * z + complex(x)
        v = abs(z)

        if hlif == "det":
            fired = v >= theta
        else:
            fired = bool(fire_fn(np.array([v]), theta, hlif_params, rng)[0])

        if fired:
            spikes[t] = True
            z = 0.0 + 0.0j
            cooldown = refr

    return spikes
