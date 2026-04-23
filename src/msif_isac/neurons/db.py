"""DB-LIF. Double-barrier bistable membrane.

Paper Section 4.4, Eqs. (6)-(7):

    tau dv/dt = -v + I(t)

with bistable state dynamics:

    state = high  if v >= theta_high and state was low
    state = low   if v <= theta_low  and state was high
    state unchanged otherwise

Biological mapping: electric-fish jamming-avoidance response (JAR)
push-pull circuit, where two prepacemaker nuclei drive frequency up
(AMPA) vs. down (NMDA). The DB-LIF abstracts this into a single
bistable membrane.

The DB-LIF ranks second across FMCW (d=1.949) and Xiangyu (d~1.82),
demonstrating that dual-timescale and bistable architectures are two
effective routes to envelope-dominated radar discrimination.
"""

from __future__ import annotations

import numpy as np

from msif_isac import thresholds as th


def run(
    signal: np.ndarray,
    beta_m: float = 0.9,
    theta_high: float = 0.15,
    theta_low: float = 0.05,
    refr: int = 3,
    hlif: str = "det",
    hlif_params=None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run a DB-LIF on the input signal.

    Returns
    -------
    spikes : state-transition events. True at the index where the state
             switches low->high (fired) or high->low (reset). This binary
             transition stream is what the paper uses as the discrimination
             feature.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if np.iscomplexobj(signal):
        signal = np.abs(signal)

    fire_fn, param_cls = th.get_threshold(hlif)
    if hlif_params is None and hlif != "det":
        hlif_params = param_cls()

    v = 0.0
    state_high = False
    cooldown = 0
    spikes = np.zeros_like(signal, dtype=bool)

    for t, x in enumerate(signal):
        if cooldown > 0:
            cooldown -= 1
            v *= beta_m
            continue

        v = beta_m * v + float(x)

        if not state_high:
            # Looking for an upward crossing of theta_high.
            if hlif == "det":
                fired = v >= theta_high
            else:
                fired = bool(fire_fn(np.array([v]), theta_high, hlif_params, rng)[0])
            if fired:
                spikes[t] = True
                state_high = True
                cooldown = refr
        else:
            # In the high state. Wait for v to fall below theta_low.
            if v <= theta_low:
                spikes[t] = True  # reset transition also counts as an event
                state_high = False
                cooldown = refr

    return spikes
