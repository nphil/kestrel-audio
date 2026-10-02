"""Turning audio into Perch windows and Perch logits into per-species confidence curves (no model in here)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .codec import PERCH_SR, SR, to_perch_rate
from .locate import HOP, WIN, window_starts

MIN_CONF = 0.0002                      # confidences below this count as zero (the evaluation's tool never reported them)
WINDOW_SAMPLES = int(WIN * PERCH_SR)   # 160000
HOP_SAMPLES = int(HOP * PERCH_SR)      # 16000


@dataclass(frozen=True)
class WindowScores:
    starts: np.ndarray      # window start times (s)
    conf: np.ndarray        # confidence of the species being looked for, per window
    any_conf: np.ndarray    # highest confidence of ANY species, per window

    def best(self) -> float:
        return float(self.conf.max()) if self.conf.size else 0.0

    def best_any(self) -> float:
        return float(self.any_conf.max()) if self.any_conf.size else 0.0


class Scorer(Protocol):
    """Anything that can score audio (22.05 kHz float32) with Perch."""

    def score(self, arrays: Sequence[np.ndarray], species_idx: int, *, full_only: bool = False) -> list[WindowScores]:
        ...


def make_windows(x22: np.ndarray, *, full_only: bool) -> tuple[list[float], np.ndarray]:
    """Cut a 22.05 kHz array into Perch windows (5 s at 32 kHz, 0.5 s hop). Returns (start times, [n, 160000] float32).
    The last window may be zero-padded. An array shorter than a window gives a single padded window."""
    x32 = to_perch_rate(x22)
    starts = window_starts(len(x22) / SR, full_only=full_only) or [0.0]
    out = np.zeros((len(starts), WINDOW_SAMPLES), dtype=np.float32)
    for i, s in enumerate(starts):
        a = int(round(s * PERCH_SR))
        seg = x32[a:a + WINDOW_SAMPLES]
        out[i, :len(seg)] = seg
    return starts, out


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def to_scores(starts: Sequence[float], logits: np.ndarray, species_idx: int) -> WindowScores:
    p = softmax(logits)
    conf = p[:, species_idx]
    anyc = p.max(axis=1)
    conf = np.where(conf >= MIN_CONF, conf, 0.0)
    anyc = np.where(anyc >= MIN_CONF, anyc, 0.0)
    return WindowScores(np.asarray(starts, dtype=np.float64), conf, anyc)
