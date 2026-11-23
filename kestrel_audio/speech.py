"""Human voices in the background of a clip.

Two listeners have to agree before anything is touched: a voice-activity model (Silero VAD) says WHEN someone is
speaking, to 32 ms, and Perch's own sound-event classes say THAT a human voice is in the 5 s window around it. Either
alone is wrong too often (the voice model takes some bird song for speech; Perch hears "singing" in an owl's trill).
What is then done about the voice is a clean-up like any other: it ships only if Perch is as sure of the animal as before.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.signal import resample_poly

from .codec import SR
from .locate import WIN
from .scoring import Survey, softmax
from .species import Labels

VAD_FILE = "silero_vad.onnx"   # Silero VAD v6 (MIT), 2 MB
VAD_SR = 16000
VAD_CHUNK = 512                # the model answers once per 32 ms ...
VAD_CONTEXT = 64               # ... and wants the last 4 ms of the chunk before
VAD_LEVEL = 0.5                # what the loudest half percent of samples is scaled to (the clips are recorded ~100x too quiet)

ON, OFF = 0.5, 0.35            # a voice starts at this probability and lasts until it has stayed under the lower one ...
JOIN_S = 0.4                   # ... for this long (the pauses between words belong to the sentence)
MIN_SPAN_S = 0.25              # shorter than this is not a word
PAD_BEFORE_S, PAD_AFTER_S = 0.15, 0.25    # soft onsets and trailing breath are part of it
MIN_SHARE = 0.12               # Perch agrees when this share of its sound-event belief is a human voice (or a voice class leads)
MIN_OVERLAP_S = 0.25           # ... in a window that holds at least this much of the voice

VOICE_EVENTS = (               # Perch v2's sound-event classes that are a human voice; names it does not have are skipped
    "Speech", "Human_voice", "Conversation", "Chatter", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
    "Child_speech_and_kid_speaking", "Narration_and_monologue", "Speech_synthesizer", "Whispering", "Shout", "Yell",
    "Children_shouting", "Laughter", "Giggle", "Chuckle_and_chortle", "Singing", "Male_singing", "Female_singing",
    "Child_singing", "Babbling", "Crowd", "Hubbub_and_speech_noise_and_speech_babble", "Cheering",
)


def voice_classes(labels: Labels) -> np.ndarray:
    """Perch class indices of the human-voice sound events, ascending."""
    found = {labels.find(name) for name in VOICE_EVENTS}
    return np.array(sorted(i for i in found if i is not None), dtype=np.int64)


class SileroVad:
    """Speech probability for every 32 ms of audio. A small recurrent net, always on the CPU (about 0.1 s per moment)."""

    def __init__(self, path: str | Path, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def __call__(self, x22: np.ndarray) -> np.ndarray:
        x = resample_poly(np.asarray(x22, dtype=np.float64), 320, 441).astype(np.float32)      # 22.05 kHz -> 16 kHz
        level = float(np.percentile(np.abs(x), 99.5)) if x.size else 0.0
        if level > 0.0:
            x = np.clip(x * (VAD_LEVEL / level), -1.0, 1.0)
        n = len(x) // VAD_CHUNK
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, VAD_CONTEXT), dtype=np.float32)
        rate = np.array(VAD_SR, dtype=np.int64)
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            chunk = np.concatenate([context, x[None, i * VAD_CHUNK:(i + 1) * VAD_CHUNK]], axis=1)
            prob, state = self.session.run(None, {"input": chunk, "state": state, "sr": rate})
            context = chunk[:, -VAD_CONTEXT:]
            out[i] = prob[0, 0]
        return out


def spans_from(prob: Sequence[float], *, hop_s: float = VAD_CHUNK / VAD_SR, total_s: float | None = None) -> list[tuple[float, float]]:
    """Speech probabilities -> the stretches (seconds) a voice is heard in, padded and merged."""
    raw: list[tuple[int, int]] = []
    start, end = None, 0
    for i, p in enumerate(prob):
        if start is None:
            if p >= ON:
                start, end = i, i + 1
        elif p >= OFF:
            end = i + 1
        elif (i + 1 - end) * hop_s >= JOIN_S:
            raw.append((start, end))
            start = None
    if start is not None:
        raw.append((start, end))
    limit = len(prob) * hop_s if total_s is None else total_s
    out: list[tuple[float, float]] = []
    for a, b in raw:
        if (b - a) * hop_s < MIN_SPAN_S:
            continue
        lo, hi = max(0.0, a * hop_s - PAD_BEFORE_S), min(limit, b * hop_s + PAD_AFTER_S)
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def voice_share(logits: np.ndarray, events: np.ndarray, voices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per window: the share of Perch's sound-event belief that is a human voice, and whether a voice class leads the events."""
    belief = softmax(np.asarray(logits)[:, events])
    at = np.searchsorted(events, voices)
    return belief[:, at].sum(axis=1), np.isin(belief.argmax(axis=1), at)


@dataclass(frozen=True)
class Voice:
    """A human voice heard in a matched moment."""
    spans: tuple[tuple[float, float], ...]     # where, in seconds from the start of the moment
    seconds: float
    share: float                               # Perch's voice share in the windows around it


@dataclass(frozen=True)
class SpeechDetector:
    vad: Callable[[np.ndarray], np.ndarray]    # 22.05 kHz audio -> speech probability per 32 ms
    events: np.ndarray                         # Perch class indices of every sound event, ascending
    voices: np.ndarray                         # ... of the human-voice ones among them

    def find(self, xs: np.ndarray, seg_start: float, survey: Survey | None) -> Voice | None:
        """The voice in the moment `xs` (which starts `seg_start` seconds into the clip `survey` covers), if both listeners hear one."""
        if survey is None or not len(survey.starts) or not len(self.voices):
            return None
        spans = spans_from(self.vad(xs), total_s=len(xs) / SR)
        if not spans:
            return None
        share, leads = voice_share(survey.logits, self.events, self.voices)
        near = np.array([any(min(s + WIN, seg_start + b) - max(s, seg_start + a) >= MIN_OVERLAP_S for a, b in spans) for s in survey.starts])
        if not near.any():
            return None
        best = float(share[near].max())
        if best < MIN_SHARE and not bool(leads[near].any()):
            return None
        return Voice(tuple(spans), float(sum(b - a for a, b in spans)), best)
