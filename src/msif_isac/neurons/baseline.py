"""Baseline leaky integrate-and-fire.

A single membrane with no specialized feature extractor. The paper uses this
as a control to separate the MS-IF feature axis from the H-LIF threshold axis,
and also to document the "oilbird effect" (Section 5.3): when the sensing
task is simple, a generic temporal integrator suffices.
"""

from __future__ import annotations

import numpy as np

from msif_isac import thresholds as th


def run(
    signal: np.ndarray,
    beta_m: float = 0.9,
    theta: float = 0.15,
    refr: int = 3,
    hlif: str = "det",
    hlif_params=None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Integrate a signal through a baseline LIF and return the spike train.

    Parameters
    ----------
    signal : 1-D array of drive samples (magnitude recommended for complex input).
    beta_m : leak coefficient per timestep (membrane decay).
    theta  : base firing threshold.
    refr   : refractory period in timesteps.
    hlif   : name of the H-LIF threshold variant ('det', 'stoch', 'bh', 'th').
    hlif_params : dataclass instance for the selected variant (None = paper defaults).
    rng    : numpy Generator (for stochastic thresholds).

    Returns
    -------
    spikes : boolean array, same length as signal.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if np.iscomplexobj(signal):
        signal = np.abs(signal)

    fire_fn, param_cls = th.get_threshold(hlif)
    if hlif_params is None and hlif != "det":
        hlif_params = param_cls()

    v = 0.0
    cooldown = 0
    spikes = np.zeros_like(signal, dtype=bool)

    for t, x in enumerate(signal):
        if cooldown > 0:
            cooldown -= 1
            v *= beta_m  # continue decay during refractory window
            continue

        v = beta_m * v + float(x)

        if hlif == "det":
            fired = v >= theta
        else:
            fired = bool(fire_fn(np.array([v]), theta, hlif_params, rng)[0])

        if fired:
            spikes[t] = True
            v = 0.0
            cooldown = refr

    return spikes
