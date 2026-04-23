"""Dual-τ LIF. Two parallel membranes with separated time constants.

Paper Section 4.1, Eqs. (1)-(2):

    tau_fast dv_fast/dt = -v_fast + I(t),  spike if v_fast >= theta
    tau_slow dv_slow/dt = -v_slow + I(t),  spike if v_slow >= theta

where tau_fast ~ O(mu s) matches the symbol rate and tau_slow ~ O(ms) matches
Doppler timescales. Biological mapping: FM bat FM-FM delay-line circuit.
The two membranes produce two independent spike channels. The paper reports
the combined spike count as the discrimination feature. This implementation
returns both channels so downstream code can use either.

The Dual-τ LIF wins two of three datasets in the paper: FMCW (d=1.957)
and Xiangyu (d=1.985). Section 5.3 discusses why.
"""

from __future__ import annotations

import numpy as np

from msif_isac import thresholds as th


def run(
    signal: np.ndarray,
    tau_fast: float = 2.0,
    tau_slow: float = 40.0,
    theta: float = 0.15,
    refr: int = 3,
    hlif: str = "det",
    hlif_params=None,
    rng: np.random.Generator | None = None,
    return_channels: bool = False,
) -> np.ndarray:
    """Run a Dual-τ LIF on the input signal.

    Parameters
    ----------
    signal : 1-D array. If complex, magnitude is taken.
    tau_fast, tau_slow : discrete time constants in timesteps.
    theta  : firing threshold (same for both membranes).
    refr   : refractory period in timesteps.
    hlif   : H-LIF threshold variant ('det'|'stoch'|'bh'|'th').
    hlif_params : parameter dataclass; None = paper defaults.
    rng    : numpy Generator for stochastic thresholds.
    return_channels : if True, return (fast_spikes, slow_spikes) stacked.

    Returns
    -------
    spikes : 1-D boolean array of combined spikes (OR across channels) by default,
             or a 2-D array (2, N) with fast and slow channels stacked.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if np.iscomplexobj(signal):
        signal = np.abs(signal)

    # Discrete-time equivalents. beta = exp(-1/tau).
    beta_fast = np.exp(-1.0 / tau_fast)
    beta_slow = np.exp(-1.0 / tau_slow)

    fire_fn, param_cls = th.get_threshold(hlif)
    if hlif_params is None and hlif != "det":
        hlif_params = param_cls()

    v_f = 0.0
    v_s = 0.0
    cool_f = 0
    cool_s = 0
    n = len(signal)
    out = np.zeros((2, n), dtype=bool)

    for t, x in enumerate(signal):
        # Fast channel
        if cool_f > 0:
            cool_f -= 1
            v_f *= beta_fast
        else:
            v_f = beta_fast * v_f + float(x)
            if hlif == "det":
                fired = v_f >= theta
            else:
                fired = bool(fire_fn(np.array([v_f]), theta, hlif_params, rng)[0])
            if fired:
                out[0, t] = True
                v_f = 0.0
                cool_f = refr

        # Slow channel
        if cool_s > 0:
            cool_s -= 1
            v_s *= beta_slow
        else:
            v_s = beta_slow * v_s + float(x)
            if hlif == "det":
                fired = v_s >= theta
            else:
                fired = bool(fire_fn(np.array([v_s]), theta, hlif_params, rng)[0])
            if fired:
                out[1, t] = True
                v_s = 0.0
                cool_s = refr

    if return_channels:
        return out
    return out[0] | out[1]
