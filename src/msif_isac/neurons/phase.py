"""Phase-LIF. Complex-valued membrane with phase-deviation spiking.

Paper Section 4.3, Eqs. (4)-(5):

    tau dz/dt = -z + I(t) * exp(-j * 2*pi * f_ref * t)
    |Delta phi(t)| = |phi(t) - phi(t - Delta t)| >= phi_th

The signal is first heterodyned by a reference frequency, then a complex
LIF tracks the analytic phase. The neuron fires when the instantaneous
phase deviation over Delta t exceeds phi_th.

Biological mapping: dolphin temporal-fine-structure (TFS) processing within
a ~250 us window, with 1-2 us phase sensitivity (Christman et al. 2025,
Finneran et al. 2019).

The Phase-LIF wins DroneRF in the paper (d=0.616). Section 5.5 discusses
why its sensitivity to rapid temporal transients generalizes beyond the
narrow dolphin-inspired use case.
"""

from __future__ import annotations

import numpy as np

from msif_isac import thresholds as th


def run(
    signal: np.ndarray,
    f_ref_norm: float = 0.0,
    tau: float = 10.0,
    dt_compare: int = 1,
    phi_th: float = 0.15,
    refr: int = 3,
    hlif: str = "det",
    hlif_params=None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run a Phase-LIF on the input signal.

    Parameters
    ----------
    signal : 1-D array (real or complex). If real, an analytic signal is
             built via the Hilbert transform.
    f_ref_norm : reference frequency for heterodyning (cycles/sample).
                 Set to 0 to read the analytic signal directly.
    tau    : membrane time constant in timesteps.
    dt_compare : phase comparison lag in timesteps (biological analog of
                 the 1-2 us echo-delay jitter threshold).
    phi_th : phase-deviation threshold in radians.
    refr, hlif, hlif_params, rng : as elsewhere.
    """
    from scipy.signal import hilbert

    signal = np.asarray(signal)
    if not np.iscomplexobj(signal):
        signal = hilbert(signal.astype(np.float64)).astype(np.complex128)

    n = signal.shape[0]
    t_idx = np.arange(n)
    if f_ref_norm != 0.0:
        mixer = np.exp(-1j * 2.0 * np.pi * f_ref_norm * t_idx)
        signal = signal * mixer

    beta = np.exp(-1.0 / tau)

    fire_fn, param_cls = th.get_threshold(hlif)
    if hlif_params is None and hlif != "det":
        hlif_params = param_cls()

    z = 0.0 + 0.0j
    z_hist = np.zeros(n, dtype=np.complex128)
    cooldown = 0
    spikes = np.zeros(n, dtype=bool)

    # First pass: evolve the complex LIF and store its trajectory.
    for t, x in enumerate(signal):
        z = beta * z + complex(x)
        z_hist[t] = z

    # Second pass: compute phase deviations and fire.
    phi = np.angle(z_hist)
    for t in range(n):
        if cooldown > 0:
            cooldown -= 1
            continue
        if t < dt_compare:
            continue
        dphi = np.angle(np.exp(1j * (phi[t] - phi[t - dt_compare])))
        v = abs(dphi)

        if hlif == "det":
            fired = v >= phi_th
        else:
            fired = bool(fire_fn(np.array([v]), phi_th, hlif_params, rng)[0])

        if fired:
            spikes[t] = True
            cooldown = refr

    return spikes
