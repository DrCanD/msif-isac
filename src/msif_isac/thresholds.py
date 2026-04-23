"""Hazard-Based LIF (H-LIF) threshold functions.

Four variants are used in the paper (Section 5.1):

- Det: deterministic hard threshold. Zero free parameters.
- Stoch: additive Gaussian noise on the comparison.
- BH: Barrier-Hazard. Dual-pathway hazard with classical sigmoid and
  barrier-crossing term.
- TH: Tunneling-Hazard. Soft threshold with a tunneling-dominant pathway
  enabling sub-threshold firing.

Each function takes the membrane potential v and threshold theta and returns
a boolean spike flag (Det) or a per-step firing probability (Stoch, BH, TH).

H-LIF parameter defaults mirror the paper: sigma=0.08, lambda_max/kappa/eta/delta
as documented below. These are fixed across all datasets.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# Parameter dataclasses
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class DetParams:
    """Deterministic threshold. No parameters."""
    pass


@dataclass(frozen=True)
class StochParams:
    """Stochastic threshold: v + sigma*N(0,1) >= theta."""
    sigma: float = 0.08


@dataclass(frozen=True)
class BHParams:
    """Barrier-Hazard threshold. Dual-pathway: classical + barrier."""
    lambda_max: float = 3.0
    kappa: float = 5.0
    eta: float = 3.0
    delta: float = 0.1


@dataclass(frozen=True)
class THParams:
    """Tunneling-Hazard threshold. Tunneling-dominant soft threshold."""
    lambda_max: float = 0.5
    kappa: float = 1.0
    eta: float = 3.0
    delta: float = 0.02


# ----------------------------------------------------------------------------
# Threshold functions
# ----------------------------------------------------------------------------

def det(v: np.ndarray, theta: float, _: DetParams | None = None) -> np.ndarray:
    """Deterministic threshold: fires when v >= theta."""
    return v >= theta


def stoch(
    v: np.ndarray,
    theta: float,
    params: StochParams | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Stochastic threshold: v + sigma*N(0,1) >= theta."""
    params = params or StochParams()
    rng = rng or np.random.default_rng()
    noise = rng.standard_normal(np.shape(v)) * params.sigma
    return (v + noise) >= theta


def _barrier_hazard_prob(
    v: np.ndarray,
    theta: float,
    lambda_max: float,
    kappa: float,
    eta: float,
    delta: float,
) -> np.ndarray:
    """Shared hazard probability kernel used by BH and TH.

    Two exponential pathways combined additively:

      lambda(v) = lambda_max * [ sigmoid(kappa*(v-theta)) + exp(-eta*|v-theta|) * delta ]

    The first term is the classical sigmoid crossing. The second is a
    sub-threshold tunneling contribution. BH uses large lambda_max/kappa;
    TH suppresses the classical pathway and keeps the tunneling term gentle.
    """
    x = v - theta
    # Classical sigmoid pathway. Clipped for numerical stability.
    sig = 1.0 / (1.0 + np.exp(np.clip(-kappa * x, -60, 60)))
    # Tunneling pathway: sub-threshold probability decays with |x|.
    tun = delta * np.exp(np.clip(-eta * np.abs(x), -60, 60))
    prob = lambda_max * (sig + tun)
    return np.clip(prob, 0.0, 1.0)


def bh(
    v: np.ndarray,
    theta: float,
    params: BHParams | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Barrier-Hazard firing decision.

    Note: BH is known to saturate on FMCW/DroneRF because both pathways
    fire aggressively (Section 5.4 Finding 3, ~2.7x Det firing rate).
    This is the intended behaviour; the saturation is what the paper documents.
    """
    params = params or BHParams()
    rng = rng or np.random.default_rng()
    prob = _barrier_hazard_prob(
        v, theta, params.lambda_max, params.kappa, params.eta, params.delta
    )
    return rng.random(np.shape(v)) < prob


def th(
    v: np.ndarray,
    theta: float,
    params: THParams | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Tunneling-Hazard firing decision.

    Same functional form as BH but with suppressed classical pathway
    (lambda_max=0.5, kappa=1) and gentle tunneling. Produces ~1.2x Det
    firing rate on a standard test signal (Section 5.4 Finding 3).
    """
    params = params or THParams()
    rng = rng or np.random.default_rng()
    prob = _barrier_hazard_prob(
        v, theta, params.lambda_max, params.kappa, params.eta, params.delta
    )
    return rng.random(np.shape(v)) < prob


# ----------------------------------------------------------------------------
# Registry for string-based lookup
# ----------------------------------------------------------------------------

REGISTRY = {
    "det": (det, DetParams),
    "stoch": (stoch, StochParams),
    "bh": (bh, BHParams),
    "th": (th, THParams),
}


def get_threshold(name: str):
    """Return the (function, default_params_class) pair for a named variant."""
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown H-LIF threshold '{name}'. Available: {list(REGISTRY)}"
        )
    return REGISTRY[key]


def is_stochastic(name: str) -> bool:
    """True for thresholds that need multi-run averaging."""
    return name.lower() in {"stoch", "bh", "th"}
