"""Clean-ups that only turn things down: outside the pitch range the animal uses, between its calls, and where a human
voice is heard. Nothing is re-synthesised and no time-frequency cell gets a gain of its own, so these cannot leave the
watery residue a spectral gate or a separator can. Like every clean-up they still have to pass the Perch re-check.

All of them are one gain map over the moment's spectrogram (a gain per pitch, times a gain per frame), applied in one
pass. All at 22.05 kHz, pure numpy/scipy.

The pitch range takes two witnesses. The signal says where the calls' own energy lies (`call_range`: the loud frames
minus the quiet ones), and no cut may reach into that. Perch says how far outside it a cut may go: the low end is cut
away rung by rung, then the high end, and the band stops where Perch starts to lose the animal (`rungs_within`,
`band_probes`, `bands_from`). Where the calls do not stand clear of the background, nothing is cut at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from . import dsp

Band = tuple[float, float]     # Hz; 0 = no lower edge, inf = no upper edge

HIGH_PASS_HZ = (500.0, 900.0, 1500.0, 2400.0)       # the rungs tried from below ...
LOW_PASS_HZ = (8500.0, 6500.0, 5000.0, 3800.0)      # ... and from above
BAND_SEARCH_LOSS = 0.03        # a rung is fine while Perch keeps all but 3% of its confidence (the finished candidate is checked strictly)
CALL_SHARE = 0.01              # the calls' own range leaves out this share of their energy at each end ...
CALL_MARGIN = 1.2              # ... and a cut starts no nearer than this factor from it
BAND_EDGE = 1.25               # a cut is complete this factor past the edge
BAND_DEPTH_DB = -30.0          # how far everything outside is turned down
MIN_CALL_RANGE_DB = 6.0        # the calls must stand at least this far above the background for "between calls" to mean anything
OPEN_ABOVE_FLOOR_DB = 2.0      # fully hushed up to this far above the background level ...
OPEN_SHARE = 0.4               # ... untouched from this share of the way up to the calls (and at least OPEN_MIN_DB above the background)
OPEN_MIN_DB = 5.0
HOLD_FRAMES = 3                # the hush lifts this many frames (35 ms) before a call and stays lifted as long after ...
RELEASE_S = 0.15               # ... then closes over about this long
DUCK_DB = -40.0                # a voice span is turned down this far
DUCK_RAMP_FRAMES = 5           # over this many frames (58 ms) at each end
VOICE_BODY_HZ = 900.0          # most of a voice lies below this: a band that starts lower would spare the voice with the animal

DUCK_FULL, DUCK_OUTSIDE = "full", "outside"    # the whole span, or only the pitches outside the animal's band


@dataclass
class Scene:
    """What the shaping needs to know about one matched moment."""
    Z: np.ndarray              # its spectrogram (complex STFT)
    f: np.ndarray              # bin frequencies (Hz)
    n: int                     # its length in samples
    voiced: np.ndarray         # per frame: a human voice is being heard
    narrow: Band | None = None  # the animal's pitch range as far as Perch lets it be cut (set once the ladder has been run)
    wide: Band | None = None    # the same, one rung more careful on each side

    @property
    def free(self) -> np.ndarray:
        """Frames no voice is heard in: the ones the clarity figures are taken over."""
        return ~self.voiced


def frames_in(spans: Sequence[tuple[float, float]], nt: int) -> np.ndarray:
    """Which of `nt` spectrogram frames (centred every HOP samples) fall inside the time spans (seconds)."""
    t = np.arange(nt) * dsp.HOP / dsp.SR
    out = np.zeros(nt, dtype=bool)
    for a, b in spans:
        out |= (t >= a) & (t <= b)
    return out


def analyse(xs: np.ndarray, spans: Sequence[tuple[float, float]] = ()) -> Scene:
    """Look at one moment (`xs`, pre-scaled). `spans` are the times a human voice is heard in it."""
    f, _, Z = dsp._stft(xs)
    return Scene(Z, f, len(xs), frames_in(spans, Z.shape[1]))


def band_gain(f: np.ndarray, band: Band) -> np.ndarray:
    """Gain per bin: 1 inside the band, BAND_DEPTH_DB outside, a raised-cosine slope (in pitch) between."""
    floor = 10 ** (BAND_DEPTH_DB / 20.0)
    lo, hi = band
    g = np.ones(len(f))
    ff = np.maximum(f, 1e-6)
    if lo > 0.0:
        x = np.clip(np.log(ff * BAND_EDGE / lo) / np.log(BAND_EDGE), 0.0, 1.0)     # 0 at lo / BAND_EDGE, 1 at lo
        g *= floor + (1.0 - floor) * (0.5 - 0.5 * np.cos(np.pi * x))
    if np.isfinite(hi):
        x = np.clip(np.log(hi * BAND_EDGE / ff) / np.log(BAND_EDGE), 0.0, 1.0)     # 1 at hi, 0 at hi * BAND_EDGE
        g *= floor + (1.0 - floor) * (0.5 - 0.5 * np.cos(np.pi * x))
    return g


def call_range(scene: Scene) -> Band | None:
    """Where the calls' own energy lies (Hz): the spectrum of the loudest tenth of the voice-free frames minus that of the
    quietest three tenths, between the points that leave CALL_SHARE of it out at each end, widened by CALL_MARGIN.
    None when the calls do not stand clear of the background: nothing in the signal says where they are then."""
    P = np.abs(scene.Z) ** 2
    level = dsp.band_ms_db(P, scene.f, dsp.LOW_HZ, dsp.HIGH_HZ)
    free = scene.free if int(scene.free.sum()) >= 10 else np.ones(len(level), dtype=bool)
    floor, top = float(np.percentile(level[free], 20)), float(np.percentile(level[free], 95))
    if top - floor < MIN_CALL_RANGE_DB:
        return None
    loud, quiet = free & (level >= np.percentile(level[free], 90)), free & (level <= np.percentile(level[free], 30))
    own = np.maximum(P[:, loud].mean(axis=1) - P[:, quiet].mean(axis=1), 0.0)
    own[scene.f < dsp.LOW_HZ / 2] = 0.0             # rumble that happens to swell with the calls is not the animal
    total = float(own.sum())
    if total <= 0.0:
        return None
    share = np.cumsum(own) / total
    last = len(scene.f) - 1
    lo = float(scene.f[min(int(np.searchsorted(share, CALL_SHARE)), last)])
    hi = float(scene.f[min(int(np.searchsorted(share, 1.0 - CALL_SHARE)), last)])
    return lo / CALL_MARGIN, hi * CALL_MARGIN


def rungs_within(keep: Band) -> tuple[list[float], list[float]]:
    """The rungs that stay clear of the calls' own range `keep`: (cuts from below, cuts from above), each in ladder order."""
    return [r for r in HIGH_PASS_HZ if r <= keep[0]], [r for r in LOW_PASS_HZ if r >= keep[1]]


def band_probes(scene: Scene, lows: Sequence[float], highs: Sequence[float]) -> list[np.ndarray]:
    """The moment with one end cut away, once per rung: first the low end (`lows`), then the high end (`highs`)."""
    return ([render(scene, band=(lo, float("inf"))) for lo in lows] + [render(scene, band=(0.0, hi)) for hi in highs])


def bands_from(lows: Sequence[float], highs: Sequence[float], fine: Sequence[bool]) -> tuple[Band | None, Band | None]:
    """`fine[i]`: Perch still hears the animal with probe i cut away (in `band_probes` order). Returns (narrow, wide): the
    narrow band cuts each end as far as every rung up to there was fine; the wide one stops a rung earlier on each side.
    None where that leaves nothing to cut (or, for the wide one, nothing different)."""
    def run(flags: Sequence[bool]) -> int:
        n = 0
        for ok in flags:
            if not ok:
                break
            n += 1
        return n

    def band(a: int, b: int) -> Band | None:
        return None if a == 0 and b == 0 else (lows[a - 1] if a else 0.0, highs[b - 1] if b else float("inf"))

    a, b = run(fine[:len(lows)]), run(fine[len(lows):])
    narrow, wide = band(a, b), band(max(0, a - 1), max(0, b - 1))
    return narrow, (wide if wide != narrow else None)


def presence(scene: Scene, band: Band | None) -> np.ndarray:
    """Per frame, 0..1: is a call sounding? A downward expander's side chain: the level inside the band against the level
    the background sits at (the quietest fifth of the voice-free frames). Opens 35 ms early, holds 35 ms, then lets go over
    about 150 ms. All ones when the calls do not stand clear of the background (a chorus, a steady trill): there is no
    "between" to hush."""
    lo, hi = (dsp.LOW_HZ, dsp.HIGH_HZ) if band is None else (max(band[0], dsp.LOW_HZ / 2), min(band[1], dsp.HIGH_HZ))
    level = uniform_filter1d(dsp.band_ms_db(np.abs(scene.Z) ** 2, scene.f, lo, hi), size=3, mode="nearest")
    free = level[scene.free] if int(scene.free.sum()) >= 10 else level
    floor, top = float(np.percentile(free, 20)), float(np.percentile(free, 95))
    if top - floor < MIN_CALL_RANGE_DB:
        return np.ones(len(level))
    start = floor + OPEN_ABOVE_FLOOR_DB
    full = floor + max(OPEN_MIN_DB, OPEN_SHARE * (top - floor))
    p = np.clip((level - start) / (full - start), 0.0, 1.0)
    p = maximum_filter1d(p, size=2 * HOLD_FRAMES + 1, mode="nearest")
    a = float(np.exp(-dsp.HOP / dsp.SR / RELEASE_S))
    r = np.empty_like(p)
    last = 0.0
    for i, v in enumerate(p):                   # a few hundred frames
        last = v if v >= last * a else last * a
        r[i] = last
    return uniform_filter1d(r, size=3, mode="nearest")


def duck_gain(scene: Scene, mode: str) -> np.ndarray | None:
    """Gain map that turns the voice spans down: everything in them (DUCK_FULL), or only what lies outside the animal's
    own pitch range (DUCK_OUTSIDE, for a voice that speaks over the call). None when that mode has nothing to act on:
    no voice, or no pitch range that leaves the body of the voice outside."""
    if not scene.voiced.any():
        return None
    floor = 10 ** (DUCK_DB / 20.0)
    d = uniform_filter1d(np.where(scene.voiced, floor, 1.0), size=DUCK_RAMP_FRAMES, mode="nearest")
    if mode == DUCK_FULL:
        return d[None, :]
    band = scene.narrow
    if band is None or band[0] < VOICE_BODY_HZ:
        return None
    edge = 10 ** (BAND_DEPTH_DB / 20.0)
    inside = ((band_gain(scene.f, band) - edge) / (1.0 - edge))[:, None]      # 1 in the band, 0 well outside
    return inside + (1.0 - inside) * d[None, :]


def render(scene: Scene, *, band: Band | None = None, hush_db: float = 0.0, duck: str | None = None) -> np.ndarray | None:
    """The moment with everything outside `band` turned down, the stretches between calls turned down by `hush_db`, and the
    voice spans ducked. None when `duck` is asked for and cannot be done."""
    G = np.ones((1, scene.Z.shape[1]))
    if band is not None:
        G = band_gain(scene.f, band)[:, None] * G
    if hush_db > 0.0:
        G = G * (10 ** (-hush_db * (1.0 - presence(scene, band)) / 20.0))[None, :]
    if duck is not None:
        d = duck_gain(scene, duck)
        if d is None:
            return None
        G = G * d
    return dsp._istft(scene.Z * G, scene.n)


def duck(v: np.ndarray, scene: Scene, mode: str) -> np.ndarray:
    """Any other candidate for the same moment (`v`) with the voice spans ducked the same way."""
    d = duck_gain(scene, mode)
    if d is None:
        return v
    return dsp._istft(dsp._stft(v)[2] * d, len(v))
