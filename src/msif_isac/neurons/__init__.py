"""MS-IF neuron family. Five specialized models + Baseline LIF.

Each module defines a single-neuron input-output transformation: a
biologically motivated feature extractor produces a drive signal, which
is then converted into a spike train by a LIF neuron with a specified
H-LIF threshold function.

- baseline : Standard leaky integrate-and-fire
- dual_tau : Two parallel membranes (fast + slow). Bat FM-FM delay lines.
- chirp    : Damped harmonic oscillator. Bat DSCF resonance.
- phase    : Complex-valued membrane, phase-deviation spiking. Dolphin TFS.
- db       : Double-barrier bistable membrane. Electric fish JAR push-pull.
- gabor    : Complex Gabor front-end into a LIF. Auditory cortex STRF.
"""

from msif_isac.neurons import baseline, dual_tau, chirp, phase, db, gabor

__all__ = ["baseline", "dual_tau", "chirp", "phase", "db", "gabor"]

REGISTRY = {
    "baseline": baseline.run,
    "dual_tau": dual_tau.run,
    "chirp": chirp.run,
    "phase": phase.run,
    "db": db.run,
    "gabor": gabor.run,
}


def get_neuron(name: str):
    """Return the `run` entry point of a named MS-IF model."""
    key = name.lower().replace("-", "_").replace("τ", "tau")
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown MS-IF neuron '{name}'. Available: {list(REGISTRY)}"
        )
    return REGISTRY[key]
