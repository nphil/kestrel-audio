"""Loudness, true-peak limiting, fades and the post-encode gain trim.

Everything here is plain numpy/scipy (no models, no I/O) so it can be unit-tested.

Why the pre-scale exists: the clips BirdNET-Go saves are recorded roughly 100x too quiet (-57 to -63 LUFS, some
below -70). ITU-R BS.1770 ignores everything quieter than -70 LUFS, so such a clip meters as minus infinity and a
naive "gain = target - measured" never fires. `prescale` brings the segment to a 0.9 peak BEFORE any metering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyloudnorm as pyln
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

TARGET_LUFS = -16.0
TRUE_PEAK_DB = -1.0
MAX_GAIN_REDUCTION_DB = 22.0   # past this the limiter is crushing the audio; stop and report "target not reached"

_meters: dict[int, pyln.Meter] = {}


def _meter(sr: int) -> pyln.Meter:
    m = _meters.get(sr)
    if m is None:
        m = _meters[sr] = pyln.Meter(sr)   # BS.1770-4 K-weighting + gating, filters designed for this rate
    return m


def lufs(x: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS, or -inf when it cannot be metered (too short, silent, or all gated)."""
    if x.size < int(0.4 * sr):
        return float("-inf")
    try:
        v = float(_meter(sr).integrated_loudness(np.asarray(x, dtype=np.float64)))
    except Exception:  # pyloudnorm raises on clips shorter than one gating block
        return float("-inf")
    return v if np.isfinite(v) else float("-inf")


def true_peak_db(x: np.ndarray) -> float:
    """Inter-sample ("true") peak in dBTP via 4x oversampling."""
    if x.size == 0:
        return -240.0
    up = resample_poly(np.asarray(x, dtype=np.float64), 4, 1)
    return float(20.0 * np.log10(float(np.max(np.abs(up))) + 1e-12))


def prescale(x: np.ndarray, peak: float = 0.9) -> tuple[np.ndarray, float]:
    """Scale so the sample peak is `peak`. Returns (scaled, scale). A silent input returns scale 0.0."""
    p = float(np.max(np.abs(x))) if x.size else 0.0
    if p <= 0.0 or not np.isfinite(p):
        return x.astype(np.float32), 0.0
    s = peak / p
    return (x * s).astype(np.float32), s


def fade(x: np.ndarray, sr: int, ms: float) -> np.ndarray:
    """Raised-cosine fade in and out (never longer than a quarter of the clip)."""
    y = np.asarray(x, dtype=np.float32).copy()
    n = min(int(ms * 1e-3 * sr), len(y) // 4)
    if n < 1:
        return y
    w = (0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)).astype(np.float32)
    y[:n] *= w
    y[-n:] *= w[::-1]
    return y


def _release(g: np.ndarray, a: float, block: int = 4096) -> np.ndarray:
    """Limiter release smoothing, exactly r[i] = g[i] if g[i] < r[i-1] else a*r[i-1] + (1-a)*g[i] with r[-1] = 1
    (instant attack, exponential release toward the wanted gain).

    A per-sample Python loop is far too slow for ~20 candidates per clip, so it is solved with cumulative sums: with
    u = g - r >= 0 the recursion is u[i] = max(0, a * (u[i-1] + g[i] - g[i-1])), a leaky Lindley recursion. Rescaling
    by a**(k+1) turns it into a plain running maximum of cumulative sums, done blockwise to keep the weights small.
    """
    n = len(g)
    out = np.empty(n)
    k = np.arange(block)
    apow = a ** (k + 1)
    iapow = a ** (-k.astype(np.float64))
    u_prev, g_prev = 0.0, 1.0
    for s in range(0, n, block):
        gb = g[s:s + block]
        m = len(gb)
        c = np.cumsum(np.diff(gb, prepend=g_prev) * iapow[:m])
        w = np.maximum(u_prev + c, c - np.minimum.accumulate(c))
        u = apow[:m] * np.maximum(w, 0.0)
        out[s:s + m] = gb - u
        u_prev, g_prev = u[-1], gb[-1]
    return out


def limit_true_peak(x: np.ndarray, sr: int, limit_db: float = TRUE_PEAK_DB, lookahead_ms: float = 3.0,
                    release_ms: float = 50.0) -> tuple[np.ndarray, float]:
    """Look-ahead limiter driven by a 4x-oversampled peak envelope. Returns (y, max_gain_reduction_db)."""
    lim = 10 ** (limit_db / 20.0)
    x = np.asarray(x, dtype=np.float64)
    up = np.abs(resample_poly(x, 4, 1))
    env = up[: len(x) * 4].reshape(-1, 4).max(axis=1)
    need = np.minimum(1.0, lim / np.maximum(env, 1e-12))
    if np.all(need >= 1.0):
        return x.astype(np.float32), 0.0
    half = max(2, int(lookahead_ms * 1e-3 * sr))
    g = minimum_filter1d(need, size=2 * half + 1, mode="nearest")   # already down when the peak arrives
    g = uniform_filter1d(g, size=2 * half + 1, mode="nearest")       # smooth the attack
    a = float(np.exp(-1.0 / (release_ms * 1e-3 * sr)))
    r = np.minimum(_release(g, a), need)                              # never louder than the peak allows
    y = x * r
    return y.astype(np.float32), float(-20.0 * np.log10(max(float(r.min()), 1e-6)))


@dataclass(frozen=True)
class NormInfo:
    lufs_in: float
    lufs_out: float
    gain_db: float
    max_gr_db: float
    tp_out_db: float
    note: str          # "" when clean; otherwise why the target was not met

    @property
    def clean(self) -> bool:
        return self.note == ""


def normalize(x: np.ndarray, sr: int, target: float = TARGET_LUFS, tp_db: float = TRUE_PEAK_DB, max_iter: int = 8,
              max_gr_db: float = MAX_GAIN_REDUCTION_DB) -> tuple[np.ndarray, NormInfo]:
    """Gain to `target` LUFS integrated, limited to `tp_db` dBTP. The gain is iterated so the loudness measured AFTER
    limiting lands on the target; it stops early when the limiter would need more than `max_gr_db` of reduction.
    A signal too quiet to meter at all is peak-normalised to `tp_db` instead (note says so)."""
    x = np.asarray(x, dtype=np.float64)
    l0 = lufs(x, sr)
    if not np.isfinite(l0):
        p = float(np.max(np.abs(x))) if x.size else 0.0
        if p <= 0.0:
            return x.astype(np.float32), NormInfo(l0, l0, 0.0, 0.0, true_peak_db(x), "silent")
        y, gr = limit_true_peak((x * (10 ** (tp_db / 20.0) / p)).astype(np.float32), sr, tp_db)
        return y, NormInfo(l0, lufs(y, sr), float(tp_db - 20 * np.log10(p)), gr, true_peak_db(y), "too quiet to meter; peak-normalised")
    gain = target - l0
    best = None
    for _ in range(max_iter):
        y, gr = limit_true_peak((x * 10 ** (gain / 20.0)).astype(np.float32), sr, tp_db)
        l1 = lufs(y, sr)
        best = (y, gr, l1, gain)
        err = target - l1
        if abs(err) < 0.15 or gr > max_gr_db:
            break
        gain += err
    y, gr, l1, gain = best
    if gr > max_gr_db:
        note = f"gain capped: limiter needed > {max_gr_db:.0f} dB"
    elif abs(target - l1) > 0.5:
        note = "target not reached"
    else:
        note = ""
    return y, NormInfo(l0, l1, gain, gr, true_peak_db(y), note)


def loudness_clean(info: NormInfo, target: float = TARGET_LUFS, tol: float = 0.6) -> bool:
    """A candidate may be offered to the verification step only if its normalisation was clean and on target."""
    return info.clean and abs(info.lufs_out - target) < tol


def trim_to_true_peak(y: np.ndarray, encode_decode: Callable[[np.ndarray], np.ndarray], sr: int,
                      limit_db: float = TRUE_PEAK_DB, slack_db: float = 0.02, max_iter: int = 6
                      ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """AAC overshoots the limiter ceiling by a fraction of a dB (more on sparse, heavily limited audio), so the
    true peak is measured on the DECODED file and the gain trimmed until it holds.

    `encode_decode(y)` must return the decoded audio for float input `y`. Returns (y_trimmed, decoded, trim_db, tp_db).
    """
    trim = 0.0
    dec = encode_decode(y)
    tp = true_peak_db(dec)
    for _ in range(max_iter):
        if tp <= limit_db + slack_db:
            break
        trim -= (tp - limit_db) + 0.15
        dec = encode_decode((y * 10 ** (trim / 20.0)).astype(np.float32))
        tp = true_peak_db(dec)
    return (y * 10 ** (trim / 20.0)).astype(np.float32), dec, float(trim), float(tp)
