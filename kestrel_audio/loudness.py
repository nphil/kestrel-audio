"""Loudness, true-peak limiting, fades and the post-encode gain trim.

Everything here is plain numpy/scipy (no models, no I/O) so it can be unit-tested.

Why the pre-scale exists: the clips BirdNET-Go saves are recorded roughly 100x too quiet (-57 to -63 LUFS, some
below -70). ITU-R BS.1770 ignores everything quieter than -70 LUFS, so such a clip meters as minus infinity and a
naive "gain = target - measured" never fires. `prescale` brings the segment to a 0.9 peak BEFORE any metering.

Why the target is a ceiling and not a promise: a few loud notes with quiet between them meter far below what they
sound like, so "gain until the meter says -16" drove such a moment deep into the limiter. The notes were squashed
and the hiss between them came up with the gain (measured on 172 real previews: 57 needed more than 6 dB of
limiting, 18 more than 12 dB). `normalize` therefore raises the gain toward the target only as far as the animal
keeps its dynamics and the background stays in the background; see the constants below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyloudnorm as pyln
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import butter, resample_poly, sosfilt

TARGET_LUFS = -16.0
TRUE_PEAK_DB = -1.0

FRAME_MS = 5.0                 # the level statistics below are taken over frames this long
POP_SHARE = 0.01               # the loudest 1% of frames (60 ms of a 6 s moment) can be a knock or a pop rather than the animal ...
POP_HEADROOM_DB = 3.0          # ... which is then brought down to this far above everything else before the gain is chosen ...
POP_MIN_DB = 1.0               # ... unless that would take less than this (nothing to gain)
POP_GONE = 0.05                # "the moment without its pop" keeps it at this gain (asked of Perch: did the animal go with it?)
THUMP_SHARE = 0.9              # a pop with this share of its energy under FLOOR_HZ is a thump: no animal makes so short a sound down there
CALL_LIMIT_DB = 4.0            # the limiter may shave at most this much off the peaks of the loudest sound that lasts ...
CALL_RANGE_DB = 12.0           # ... and the calls (every frame within this of that sound, the possible pop left out) ...
PUMP_TOLERANCE_DB = 0.5        # ... may not lose more than this of their energy against the background
FLOOR_HZ = 300.0               # the background is measured above the camera's rumble
FLOOR_CEILING_DB = -30.0       # the background between calls is not amplified past this (dBFS RMS) ...
FLOOR_MAX_CUT_DB = 6.0         # ... at the price of at most this much loudness

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


def _frames(x: np.ndarray, sr: int, ms: float = FRAME_MS) -> np.ndarray:
    """`x` cut into whole frames of `ms` (rows); the ragged end is dropped. At least one row."""
    n = max(1, int(ms * 1e-3 * sr))
    m = len(x) // n
    if m == 0:
        return np.asarray(x, dtype=np.float64).reshape(1, -1)
    return np.asarray(x[:m * n], dtype=np.float64).reshape(m, n)


def sustained_peak(x: np.ndarray, sr: int, pop_share: float = POP_SHARE) -> float:
    """The peak of the loudest sound that lasts: what all but the loudest `pop_share` of 5 ms frames stay under (0: the
    plain peak)."""
    if x.size == 0:
        return 0.0
    return float(np.percentile(np.abs(_frames(x, sr)).max(axis=1), 100.0 * (1.0 - pop_share)))


def pop_gain(x: np.ndarray, sr: int, headroom_db: float = POP_HEADROOM_DB, window_ms: float = 2.0) -> np.ndarray | None:
    """The gain per sample that brings an isolated knock or pop down to `headroom_db` above the loudest sound that lasts.
    None when nothing stands out by POP_MIN_DB more than that.

    Left alone, one pop decides the gain of the whole preview (everything else stays far too quiet) or, pushed into
    the limiter, drags the level down for a tenth of a second after it. This gain dips only while the pop lasts (2 ms
    either side, no release tail), so nothing around it moves. Whether the pop may be touched at all is the caller's
    question: a cardinal's chip looks exactly like this (see `without_pop`, `is_thump`)."""
    x = np.asarray(x, dtype=np.float64)
    ref = sustained_peak(x, sr)
    ceiling = ref * 10 ** (headroom_db / 20.0)
    mag = np.abs(x)
    if ref <= 0.0 or float(mag.max()) <= ceiling * 10 ** (POP_MIN_DB / 20.0):
        return None
    need = np.minimum(1.0, ceiling / np.maximum(mag, 1e-12))
    half = max(2, int(window_ms * 1e-3 * sr))
    g = minimum_filter1d(need, size=2 * half + 1, mode="nearest")   # already down when the pop arrives
    return uniform_filter1d(g, size=2 * half + 1, mode="nearest")    # and back up as smoothly as it went down


def tame(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """`x` with its pop brought down (`pop_gain`). Returns (y, the most it was turned down in dB; 0 = there was none)."""
    x = np.asarray(x, dtype=np.float64)
    g = pop_gain(x, sr)
    if g is None:
        return x, 0.0
    return x * g, float(-20.0 * np.log10(max(float(g.min()), 1e-6)))


def _pop_core(gain: np.ndarray) -> np.ndarray:
    """The samples the pop itself occupies: where its gain is down by more than half a dB."""
    return gain < 10 ** (-0.5 / 20.0)


def without_pop(x: np.ndarray, gain: np.ndarray, sr: int, pad_ms: float = 5.0) -> np.ndarray:
    """`x` with its pop all but removed (POP_GONE, 5 ms either side included). Perch is asked about this to learn whether
    the pop was the animal: a knock can go and the animal is still there, its own chip cannot."""
    k = 2 * max(1, int(pad_ms * 1e-3 * sr)) + 1
    hole = minimum_filter1d(np.where(_pop_core(gain), POP_GONE, 1.0), size=k, mode="nearest")
    return (np.asarray(x, dtype=np.float64) * uniform_filter1d(hole, size=max(3, k // 2), mode="nearest")).astype(np.float32)


def is_thump(x: np.ndarray, gain: np.ndarray, sr: int) -> bool:
    """Is the pop a low thump (wind, a knock on the housing): THUMP_SHARE of its energy under FLOOR_HZ?"""
    x = np.asarray(x, dtype=np.float64)
    core = _pop_core(gain)
    total = float((x[core] ** 2).sum())
    if total <= 0.0:
        return False
    high = sosfilt(butter(4, FLOOR_HZ, btype="highpass", fs=sr, output="sos"), x)
    return float((high[core] ** 2).sum()) <= (1.0 - THUMP_SHARE) * total


def floor_level_db(x: np.ndarray, sr: int) -> float:
    """Level of the background: what the quietest fifth of 20 ms frames reach above 300 Hz (dBFS RMS)."""
    if x.size < int(0.1 * sr):
        return float("-inf")
    y = sosfilt(butter(2, FLOOR_HZ, btype="highpass", fs=sr, output="sos"), np.asarray(x, dtype=np.float64))
    ms = (_frames(y, sr, 20.0) ** 2).mean(axis=1)
    return float(10.0 * np.log10(float(np.percentile(ms, 20)) + 1e-30))


def pump_db(before: np.ndarray, after: np.ndarray, sr: int, pop_share: float = POP_SHARE) -> float:
    """How far the calls sank against the background between `before` and `after` (dB; 0 = the same gain everywhere).
    This is the number that says the hiss came up between the notes.

    The calls are the 5 ms frames whose peak lies within CALL_RANGE_DB of the loudest sound that lasts (the loudest
    `pop_share` of frames, a possible pop, left out), taken by their total energy: a few shaved peaks hardly count, a
    call held down as a whole counts in full. The background is the middle gain change over the quietest 30% of frames."""
    n = min(len(before), len(after))
    fb, fa = _frames(before[:n], sr), _frames(after[:n], sr)
    if len(fb) < 10:
        return 0.0
    peak = np.abs(fb).max(axis=1)
    ref = float(np.percentile(peak, 100.0 * (1.0 - pop_share)))
    eb, ea = (fb ** 2).sum(axis=1) + 1e-30, (fa ** 2).sum(axis=1) + 1e-30
    calls = (peak <= ref) & (peak >= ref * 10 ** (-CALL_RANGE_DB / 20.0))
    quiet = eb <= np.percentile(eb, 30)
    if not calls.any():
        return 0.0
    return float(np.median(10.0 * np.log10(ea[quiet] / eb[quiet])) - 10.0 * np.log10(ea[calls].sum() / eb[calls].sum()))


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
    note: str                  # "" when clean; otherwise why the audio could not be made loud properly
    tamed_db: float = 0.0      # how far an isolated pop was turned down first
    held_db: float = 0.0       # how far below the target the loudness was left so the call keeps its dynamics
    floor_cut_db: float = 0.0  # ... of which this much so the background is not amplified into the foreground

    @property
    def clean(self) -> bool:
        return self.note == ""


def normalize(x: np.ndarray, sr: int, target: float = TARGET_LUFS, tp_db: float = TRUE_PEAK_DB, max_iter: int = 8, *,
              tame_pops: bool = True) -> tuple[np.ndarray, NormInfo]:
    """Make `x` loud: gain toward `target` LUFS integrated, limited to `tp_db` dBTP, and no further than the sound allows.

    The gain stops rising at whichever comes first:
      * the target loudness, measured AFTER limiting (the gain is iterated until it lands);
      * the limiter would have to shave more than CALL_LIMIT_DB off the peaks of the loudest sound that lasts, or the calls
        as a whole would sink against the background (`pump_db`: the hiss would come up between the notes);
      * the background would be amplified past FLOOR_CEILING_DB (for this one at most FLOOR_MAX_CUT_DB is given up).
    A preview that stops early is quieter on the meter but its calls already reach the ceiling.

    `tame_pops`: an isolated pop is turned down first (`pop_gain`) so it cannot decide any of this. Without it nothing
    but gain and the limiter touches the sound, and the pop is the peak everything else is measured against: for a pop
    that is the animal itself. A signal too quiet to meter at all is peak-normalised to `tp_db` instead (note says so)."""
    x = np.asarray(x, dtype=np.float64)
    l0 = lufs(x, sr)
    if not np.isfinite(l0):
        p = float(np.max(np.abs(x))) if x.size else 0.0
        if p <= 0.0:
            return x.astype(np.float32), NormInfo(l0, l0, 0.0, 0.0, true_peak_db(x), "silent")
        y, gr = limit_true_peak((x * (10 ** (tp_db / 20.0) / p)).astype(np.float32), sr, tp_db)
        return y, NormInfo(l0, lufs(y, sr), float(tp_db - 20 * np.log10(p)), gr, true_peak_db(y), "too quiet to meter; peak-normalised")
    pop = pop_gain(x, sr) if tame_pops else None
    share = POP_SHARE if tame_pops else 0.0
    xt = x if pop is None else x * pop
    tamed = 0.0 if pop is None else float(-20.0 * np.log10(max(float(pop.min()), 1e-6)))
    lt = lufs(xt, sr) if pop is not None else l0
    if not np.isfinite(lt):
        lt = l0
    want = target - lt
    cap = tp_db + CALL_LIMIT_DB - 20.0 * np.log10(max(sustained_peak(xt, sr, share), 1e-12))
    floor = floor_level_db(xt, sr)

    def render(gain: float) -> tuple[np.ndarray, float, float, float, float]:
        # give up loudness for a quieter background only as far as the total shortfall stays within FLOOR_MAX_CUT_DB
        cut = float(np.clip(floor + gain - FLOOR_CEILING_DB, 0.0, max(0.0, FLOOR_MAX_CUT_DB - max(0.0, want - gain))))
        y, gr = limit_true_peak(xt * 10 ** ((gain - cut) / 20.0), sr, tp_db)
        return y, gr, lufs(y, sr), gain - cut, cut

    gain = min(want, cap)
    best = render(gain)
    for _ in range(max_iter):
        _, gr, l1, _, cut = best
        err = target - l1
        if abs(err) < 0.15 or cut > 0.0 or gain >= cap - 1e-9:
            break
        gain = min(cap, gain + err)
        best = render(gain)
    for _ in range(6):                       # the limiter may still be holding the calls down as a whole: back off until it is not
        y, gr, l1, applied, cut = best
        if gr <= PUMP_TOLERANCE_DB:          # it never takes that much anywhere
            break
        pump = pump_db(xt * 10 ** (applied / 20.0), y, sr, share)
        if pump <= PUMP_TOLERANCE_DB:
            break
        applied -= max(0.25, pump - 0.5 * PUMP_TOLERANCE_DB)
        y, gr = limit_true_peak(xt * 10 ** (applied / 20.0), sr, tp_db)
        best = (y, gr, lufs(y, sr), applied, cut)
    y, gr, l1, applied, cut = best
    return y, NormInfo(l0, l1, applied, gr, true_peak_db(y), "", tamed, max(0.0, target - l1), cut)


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
