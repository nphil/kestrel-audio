"""Where in the clip is the animal? Pure logic over Perch's per-window scores (no model, no audio).

Perch scores 5 s windows; a 0.5 s hop gives a confidence curve for the detected species. The matched moment is the
best run of windows (each at least 90% of the best score), merged and capped at 8 s, plus a half second of margin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

WIN = 5.0          # Perch window length (s)
HOP = 0.5          # window hop (s)
MAX_SPAN = 8.0     # longest preview before it stops being "the moment"
MARGIN = 0.5       # extra audio kept on both sides of the matched run
RUN_FRACTION = 0.9
NEAR_BEST = 0.005  # windows within this of the best count as "the best" (the middle one is taken)
FALLBACK_BEGIN = 3.0   # BirdNET-Go keeps 3 s of pre-roll, so its own detection window starts 3 s into the clip
MIN_EVIDENCE = 0.10    # below this best-window confidence Perch is guessing: no moment is "matched", and nothing can be verified


@dataclass(frozen=True)
class Segment:
    start: float            # seconds into the original clip, margin included
    end: float
    best_start: float       # start of the single best 5 s window (0.0 for fallback segments)
    best_conf: float
    run_start: float        # the merged run of near-best windows, margin excluded
    run_end: float
    windows_high: int       # how many windows the run holds
    source: str             # "perch" | "fallback" | "whole"

    @property
    def length(self) -> float:
        return self.end - self.start


def window_starts(length_s: float, *, full_only: bool, win: float = WIN, hop: float = HOP) -> list[float]:
    """Window start times for a clip of `length_s`. Full windows lie wholly inside the clip. The scoring pass also uses
    windows holding at least 90% real audio (last window zero-padded), which is what the evaluation's tool did."""
    need = win if full_only else win - hop
    out, k = [], 0
    while k * hop + need <= length_s + 1e-6:
        out.append(round(k * hop, 3))
        k += 1
    return out


def pick_segment(starts: Sequence[float], conf: Sequence[float], clip_len: float, *, win: float = WIN, hop: float = HOP,
                 max_span: float = MAX_SPAN, margin: float = MARGIN) -> Segment | None:
    """The matched moment from per-window confidences (`starts`/`conf` are parallel, full windows only, ascending).
    Returns None when the species never scored (best <= 0): the caller then falls back."""
    pairs = [(round(float(s), 3), float(c)) for s, c in zip(starts, conf) if s + win <= clip_len + 1e-6]
    if not pairs:
        return None
    full = [s for s, _ in pairs]
    c = dict(pairs)
    best = max(c.values())
    if best <= 0.0:
        return None
    near = [s for s in full if c[s] >= best - NEAR_BEST]
    best_s = near[len(near) // 2]
    thr = RUN_FRACTION * best
    i = full.index(best_s)
    lo = hi = i
    while lo > 0 and c[full[lo - 1]] >= thr and abs(full[lo] - full[lo - 1] - hop) < 1e-6:
        lo -= 1
    while hi < len(full) - 1 and c[full[hi + 1]] >= thr and abs(full[hi + 1] - full[hi] - hop) < 1e-6:
        hi += 1
    run = full[lo:hi + 1]
    span = (run[0], run[-1] + win)
    if span[1] - span[0] > max_span + 1e-6:
        # too long: slide an 8 s frame over the run, keep the position holding the most confidence, break ties by
        # staying centred on the best window
        n_pos = int(round((span[1] - max_span - span[0]) / hop))
        cand = [span[0] + j * hop for j in range(n_pos + 1)]
        centre = best_s + win / 2

        def score(x: float) -> float:
            return sum(c[s] for s in run if x - 1e-9 <= s <= x + (max_span - win) + 1e-9)

        x = max(cand, key=lambda x: (round(score(x), 6), -abs(x + max_span / 2 - centre)))
        span = (float(x), float(x) + max_span)
    return Segment(start=max(0.0, span[0] - margin), end=min(clip_len, span[1] + margin), best_start=best_s,
                   best_conf=best, run_start=run[0], run_end=run[-1] + win, windows_high=len(run), source="perch")


def fallback_segment(clip_len: float, *, begin: float = FALLBACK_BEGIN, win: float = WIN, margin: float = MARGIN) -> Segment:
    """Used when the species cannot be scored (unknown name, or Perch never saw it): BirdNET-Go's own window start."""
    s = max(0.0, min(begin, max(0.0, clip_len - win)) - margin)
    e = min(clip_len, s + margin + win + margin)
    return Segment(start=s, end=e, best_start=0.0, best_conf=0.0, run_start=s, run_end=e, windows_high=0, source="fallback")


def whole_clip_segment(clip_len: float) -> Segment:
    """Clips no longer than one Perch window have nothing to search: the whole clip is the moment."""
    return Segment(start=0.0, end=clip_len, best_start=0.0, best_conf=0.0, run_start=0.0, run_end=clip_len,
                   windows_high=0, source="whole")
