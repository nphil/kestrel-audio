"""Spectral gating and the clarity metrics used to rank clean-up candidates.

All at the separator's working rate (22.05 kHz). Pure numpy/scipy.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import istft, stft
from scipy.special import exp1

SR = 22050
NFFT = 1024
HOP = 256
LOW_HZ = 300.0        # below this the camera's wind/handling rumble dominates and says nothing about the animal
HIGH_HZ = 11000.0


def _stft(x: np.ndarray):
    return stft(np.asarray(x, dtype=np.float64), SR, window="hann", nperseg=NFFT, noverlap=NFFT - HOP,
                boundary="zeros", padded=True)


def _istft(Z: np.ndarray, n: int) -> np.ndarray:
    _, y = istft(Z, SR, window="hann", nperseg=NFFT, noverlap=NFFT - HOP, boundary=True)
    y = y[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)


def noise_psd(P: np.ndarray, pct: float = 20.0) -> np.ndarray:
    """Per-bin noise power from a low percentile over time. For stationary complex-Gaussian noise the p-th percentile
    of the power is -ln(1-p) times its mean, so that factor is divided out. Robust to calls covering less than
    (100-pct)% of a bin's frames."""
    lam = np.percentile(P, pct, axis=1) / (-np.log(1.0 - pct / 100.0))
    lam = np.maximum(lam, 1e-20)
    return uniform_filter1d(lam, size=3, mode="nearest")


def clip_noise_psd(x: np.ndarray) -> np.ndarray:
    """Noise PSD of a whole clip (used for every gate strength so they all see the same noise estimate)."""
    return noise_psd(np.abs(_stft(x)[2]) ** 2)


def wiener_dd(x: np.ndarray, floor_db: float = -15.0, alpha: float = 0.98, xi_min_db: float = -25.0,
              lam: np.ndarray | None = None) -> np.ndarray:
    """Decision-directed (Ephraim-Malah) MMSE-LSA suppression with a per-bin stationary noise PSD and a gain floor.
    The decision-directed a-priori SNR estimate keeps the residual noise smooth instead of leaving isolated
    'musical' peaks. `floor_db` is the strongest cut applied to any time-frequency bin."""
    n = len(x)
    _, _, Y = _stft(x)
    P = np.abs(Y) ** 2
    if lam is None:
        lam = noise_psd(P)
    gamma = P / lam[:, None]
    nb, nt = P.shape
    G = np.ones_like(P)
    prev = np.ones(nb)
    xi_min = 10 ** (xi_min_db / 10)
    for k in range(nt):
        xi = alpha * prev + (1 - alpha) * np.maximum(gamma[:, k] - 1.0, 0.0)
        xi = np.maximum(xi, xi_min)
        g = xi / (1.0 + xi)
        v = np.maximum(g * gamma[:, k], 1e-8)
        g = np.minimum(g * np.exp(0.5 * exp1(v)), 1.0)
        prev = (g ** 2) * gamma[:, k]
        G[:, k] = g
    G = np.maximum(G, 10 ** (floor_db / 20))
    G = uniform_filter1d(G, size=3, axis=0, mode="nearest")   # smooth across frequency to avoid isolated bins
    return _istft(Y * G, n)


def band_ms_db(P: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Mean-square level (dB) per STFT frame in a band."""
    sel = (f >= lo) & (f <= hi)
    ms = 0.5 * P[sel].sum(axis=0)   # scipy 'spectrum' scaling: |Y| is the sinusoid amplitude
    return 10 * np.log10(ms + 1e-20)


def frame_sets(x: np.ndarray, loud_pct: float = 90.0, quiet_pct: float = 30.0):
    """Frame sets taken from the untouched (loudness-only) segment: the loudest 10% of frames (>= 300 Hz energy, the
    animal sounds) and the quietest 30% (background only). Returns (freqs, loud_mask, quiet_mask)."""
    f, _, Z = _stft(x)
    e = band_ms_db(np.abs(Z) ** 2, f, LOW_HZ, HIGH_HZ)
    return f, e >= np.percentile(e, loud_pct), e <= np.percentile(e, quiet_pct)


def contrast_db(x: np.ndarray) -> float:
    """Clarity: how far the loud frames stand above the quiet ones (95th minus 20th percentile of the >= 300 Hz frame
    energy, in dB). Higher means the animal stands out more against its background."""
    f, _, Z = _stft(x)
    e = band_ms_db(np.abs(Z) ** 2, f, LOW_HZ, HIGH_HZ)
    return float(np.percentile(e, 95) - np.percentile(e, 20))


def level_in_sets(x: np.ndarray, loud: np.ndarray, quiet: np.ndarray) -> tuple[float, float]:
    """Mean level (dB) of the loud-frame set and of the quiet-frame set of `x` (>= 300 Hz)."""
    f, _, Z = _stft(x)
    e = band_ms_db(np.abs(Z) ** 2, f, LOW_HZ, HIGH_HZ)
    nt = len(e)
    ld = np.pad(loud, (0, max(0, nt - len(loud))))[:nt]
    qt = np.pad(quiet, (0, max(0, nt - len(quiet))))[:nt]
    return (float(10 * np.log10(np.mean(10 ** (e[ld] / 10)) + 1e-30)),
            float(10 * np.log10(np.mean(10 ** (e[qt] / 10)) + 1e-30)))
