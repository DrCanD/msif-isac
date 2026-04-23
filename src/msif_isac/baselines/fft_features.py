"""FFT-based spectral features and a non-FFT energy baseline.

These are the six baselines reported in Table 6 of the paper. All operate
on the same input frames as the MS-IF models, so the comparison is direct.

- energy       : time-domain |x|^2 sum. No FFT.
- fft_energy   : sum of |X|^2 across the spectrum. Parseval-equivalent
                 to energy for real inputs; kept for completeness.
- fft_peak     : maximum spectral magnitude.
- fft_centroid : spectral centroid (weighted mean frequency).
- fft_flatness : geometric/arithmetic mean ratio of |X|^2. Low for
                 concentrated spectra, high for flat noise.
- fft_bandwidth: spectral bandwidth (weighted std around the centroid).
"""

from __future__ import annotations

import numpy as np


def _magnitude_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, |X|) for a real/complex signal."""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x)))
    else:
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(len(x))
    return f, np.abs(X)


def energy(x: np.ndarray) -> float:
    """Time-domain energy. No FFT."""
    x = np.asarray(x)
    return float(np.sum(np.abs(x) ** 2))


def fft_energy(x: np.ndarray) -> float:
    """Frequency-domain energy. Parseval-equivalent to time-domain for real x."""
    _, mag = _magnitude_spectrum(x)
    return float(np.sum(mag ** 2))


def fft_peak(x: np.ndarray) -> float:
    """Peak spectral magnitude."""
    _, mag = _magnitude_spectrum(x)
    return float(mag.max())


def fft_centroid(x: np.ndarray) -> float:
    """Spectral centroid: sum(f * |X|) / sum(|X|)."""
    f, mag = _magnitude_spectrum(x)
    s = mag.sum()
    if s < 1e-12:
        return 0.0
    return float(np.sum(f * mag) / s)


def fft_flatness(x: np.ndarray) -> float:
    """Spectral flatness: geometric mean / arithmetic mean of |X|^2.

    Bounded in [0, 1]. Flat spectra approach 1, tonal spectra approach 0.
    """
    _, mag = _magnitude_spectrum(x)
    p = mag ** 2
    p = p[p > 0]
    if len(p) < 2:
        return 0.0
    log_gm = np.mean(np.log(p))
    am = p.mean()
    return float(np.exp(log_gm) / (am + 1e-30))


def fft_bandwidth(x: np.ndarray) -> float:
    """Spectral bandwidth: weighted std of frequency around the centroid."""
    f, mag = _magnitude_spectrum(x)
    s = mag.sum()
    if s < 1e-12:
        return 0.0
    c = np.sum(f * mag) / s
    var = np.sum(((f - c) ** 2) * mag) / s
    return float(np.sqrt(max(var, 0.0)))


REGISTRY = {
    "energy": energy,
    "fft_energy": fft_energy,
    "fft_peak": fft_peak,
    "fft_centroid": fft_centroid,
    "fft_flatness": fft_flatness,
    "fft_bandwidth": fft_bandwidth,
}


def get_baseline(name: str):
    key = name.lower().replace("-", "_")
    if key not in REGISTRY:
        raise KeyError(f"Unknown baseline '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[key]
