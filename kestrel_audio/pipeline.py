"""The preview pipeline for one clip: find the matched moment, make it loud, clean it up ONLY if a Perch re-check
proves that helps, then encode. The models arrive as arguments, so the logic can be exercised with fakes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol, Sequence

import numpy as np

from . import dsp
from .candidates import METHOD_TRIM, gate_variants, method_of, separated_variants
from .codec import OUT_SR, SR, AacCodec, to_output_rate
from .locate import MIN_EVIDENCE, WIN, Segment, fallback_segment, pick_segment, whole_clip_segment
from .loudness import NormInfo, fade, loudness_clean, lufs, normalize, prescale, trim_to_true_peak
from .scoring import Scorer, Survey
from .verify import Candidate, Reference, Decision, TOLERANCE, choose, select_sources

FADE_MS = 75.0
SILENT_PEAK = 1e-5            # below -100 dBFS a clip is silence, not a quiet animal
AAC_GUARD_TOLERANCE = 0.05    # after AAC, the clean-up may score at most this far below the untouched moment


class PreviewError(RuntimeError):
    """The clip cannot be turned into a preview (silent, too short, ...)."""


class Separator(Protocol):
    k: int

    def separate(self, x: np.ndarray) -> np.ndarray:
        """x: mono float32 at 22.05 kHz -> (k, len(x)) float32."""
        ...


@dataclass
class Preview:
    audio: np.ndarray                 # what will be shipped (normalised, faded), 22.05 kHz
    reference_audio: np.ndarray       # the untouched moment, normalised: candidate B
    variant: str                      # "B" or a candidate id such as "G20"
    segment: Segment
    norm: NormInfo
    score_original: float | None      # Perch(species) on the untouched moment at preview volume
    score_preview: float | None       # ... on what will be shipped
    native_original: float | None = None
    native_preview: float | None = None
    notes: list[str] = field(default_factory=list)
    decision: list[dict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    survey: Survey | None = None      # Perch's answer for the whole clip, when the caller asked for one (see `surveyor`)

    @property
    def method(self) -> str:
        return method_of(self.variant)

    @property
    def cleaned(self) -> bool:
        return self.variant != "B"


def _trace(dec: Decision, ref: Reference, picked: str | None) -> list[dict]:
    out = []
    for c in dec.considered:
        out.append({
            "id": c.id,
            "clarityGainDb": round(c.gain_db(ref), 1),
            "backgroundCutDb": round(c.suppression_db, 1),
            "perchNative": None if c.native_score is None else round(c.native_score, 3),
            "perchLoud": None if c.loud_score is None else round(c.loud_score, 3),
            "result": "picked" if c.id == picked else (c.reason or "not scored"),
        })
    return out


def make_preview(x22: np.ndarray, *, species_idx: int | None, scorer: Scorer | None, separators: Sequence[Separator],
                 cleanup: bool = True, surveyor: Callable[[np.ndarray], Survey] | None = None) -> Preview:
    """x22: the whole clip, mono float32 at 22.05 kHz. `surveyor` runs Perch once over the whole clip and keeps every window's
    full answer; the search for the matched moment then reads its curve from that pass instead of making another, and the survey
    comes back on the Preview for whoever wants more than that one species' curve."""
    t_all = time.perf_counter()
    tm: dict[str, float] = {}
    notes: list[str] = []
    x22 = np.nan_to_num(np.asarray(x22, dtype=np.float32))
    dur = len(x22) / SR
    if dur < 0.5:
        raise PreviewError("the clip is shorter than half a second")
    if float(np.max(np.abs(x22))) < SILENT_PEAK:
        raise PreviewError("the clip is silent")

    # ---- 1. locate the matched moment
    t = time.perf_counter()
    survey = surveyor(x22) if surveyor is not None else None
    if dur <= WIN:
        seg = whole_clip_segment(dur)
    elif species_idx is not None and scorer is not None:
        ws = survey.window_scores(species_idx) if survey is not None else scorer.score([x22], species_idx, full_only=True)[0]
        seg = pick_segment(ws.starts, ws.conf, dur)
        if seg is None:
            seg = fallback_segment(dur)
            notes.append("Perch never heard the species in this clip; used BirdNET-Go's own detection window")
        elif seg.best_conf < MIN_EVIDENCE:
            # a noise-level curve has no real peak, and a clean-up cannot be checked against a score that low (the check
            # "not worse than the untouched moment minus 0.02" would pass for anything)
            notes.append(f"Perch is barely sure of the species in this clip (best {seg.best_conf:.2f}); used BirdNET-Go's own detection window and did not clean")
            seg = fallback_segment(dur)
    else:
        seg = fallback_segment(dur)
        notes.append("the species is unknown to Perch; used BirdNET-Go's own detection window")
    tm["locateS"] = time.perf_counter() - t

    x = x22[int(round(seg.start * SR)):int(round(seg.end * SR))]
    xs, scale = prescale(x)            # BEFORE any metering: very quiet clips would meter as minus infinity
    if scale == 0.0:
        raise PreviewError("the matched moment is silent")

    # ---- 2. candidate B: the moment itself, made loud
    b_final, b_norm = normalize(fade(xs, SR, FADE_MS), SR)
    can_check = seg.source == "perch" and species_idx is not None and scorer is not None
    if not can_check:
        tm["totalS"] = time.perf_counter() - t_all
        return Preview(b_final, b_final, "B", seg, b_norm, None, None, notes=notes, timings=tm, survey=survey)
    if not cleanup:
        t = time.perf_counter()
        loud = scorer.score([b_final], species_idx)[0].best()
        tm["verifyS"] = time.perf_counter() - t
        tm["totalS"] = time.perf_counter() - t_all
        return Preview(b_final, b_final, "B", seg, b_norm, loud, loud, notes=notes, timings=tm, survey=survey)

    # ---- 3. AI separation (the GPU/CPU heavy part) and one batched Perch pass over the reference and every track
    t = time.perf_counter()
    tracks = {s.k: np.asarray(s.separate(xs), dtype=np.float32) for s in separators}
    tm["separateS"] = time.perf_counter() - t

    t = time.perf_counter()
    arrays: list[np.ndarray] = [x, b_final]
    for k, tr in tracks.items():
        arrays.extend(tr[j] / scale for j in range(k))     # each track back at the moment's original volume
    res = scorer.score(arrays, species_idx)
    ref_native, ref_loud = res[0].best(), res[1].best()
    at, choices = 2, {}
    for k in tracks:
        part = res[at:at + k]
        choices[k] = select_sources([r.best() for r in part], [r.best_any() for r in part])
        if choices[k].note:
            notes.append(f"{k}-track separation: {choices[k].note}")
        at += k
    tm["trackScoreS"] = time.perf_counter() - t

    # ---- 4. build every candidate and rank it by clarity
    t = time.perf_counter()
    loud_set, quiet_set = dsp.frame_sets(b_final)[1:]
    ref = Reference(ref_native, ref_loud, dsp.contrast_db(b_final))
    _, base_quiet = dsp.level_in_sets(x, loud_set, quiet_set)
    variants = gate_variants(xs, dsp.clip_noise_psd(x22 * scale))
    for k, tr in tracks.items():
        variants.update(separated_variants(k, tr, choices[k]))
    cands: list[Candidate] = []
    native_audio: dict[str, np.ndarray] = {}
    loud_audio: dict[str, np.ndarray] = {}
    for cid, v in variants.items():
        pre = fade(v, SR, FADE_MS)
        final, info = normalize(pre, SR)
        native = pre / scale
        native_audio[cid], loud_audio[cid] = native, final
        cands.append(Candidate(cid, dsp.contrast_db(final), base_quiet - dsp.level_in_sets(native, loud_set, quiet_set)[1],
                               loudness_clean(info)))
    tm["buildS"] = time.perf_counter() - t

    # ---- 5. verify with Perch, cleanest first
    t = time.perf_counter()

    def native_scores(batch: list[Candidate]) -> dict[str, float]:
        return {c.id: r.best() for c, r in zip(batch, scorer.score([native_audio[c.id] for c in batch], species_idx))}

    def loud_scores(batch: list[Candidate]) -> dict[str, float]:
        return {c.id: r.best() for c, r in zip(batch, scorer.score([loud_audio[c.id] for c in batch], species_idx))}

    dec = choose(cands, ref, native_scores, loud_scores)
    tm["verifyS"] = time.perf_counter() - t
    tm["totalS"] = time.perf_counter() - t_all
    if dec.choice is None:
        return Preview(b_final, b_final, "B", seg, b_norm, ref_loud, ref_loud, native_original=ref_native, native_preview=ref_native,
                       notes=notes, decision=_trace(dec, ref, None), timings=tm, survey=survey)
    c = dec.choice.candidate
    return Preview(loud_audio[c.id], b_final, c.id, seg, b_norm, ref_loud, c.loud_score, native_original=ref_native,
                   native_preview=c.native_score, notes=notes, decision=_trace(dec, ref, c.id), timings=tm, survey=survey)


# ------------------------------------------------------------------------------------------------ shipping

@dataclass
class Shipped:
    data: bytes           # the .m4a
    lufs: float           # measured on the DECODED file
    true_peak_db: float
    duration_s: float
    trim_db: float        # gain trimmed so the decoded true peak holds


def ship(audio22: np.ndarray, codec: AacCodec) -> Shipped:
    """Resample to 48 kHz, AAC-encode, and trim the gain until the DECODED file's true peak is at most -1 dBTP."""
    y, dec, trim, tp = trim_to_true_peak(to_output_rate(audio22), codec.encode_decode, OUT_SR)
    return Shipped(codec.last_bytes, lufs(dec, OUT_SR), tp, len(dec) / OUT_SR, trim)


def finalize(prev: Preview, *, codec: AacCodec, scorer: Scorer | None, species_idx: int | None) -> tuple[Shipped, Preview]:
    """Encode the preview. A cleaned preview gets one last Perch check on the DECODED AAC (heavily gated audio can
    pick up codec artefacts); if it fell behind the untouched moment, the untouched moment is shipped instead."""
    t = time.perf_counter()
    shipped = ship(prev.audio, codec)
    if prev.cleaned and scorer is not None and species_idx is not None:
        plain = ship(prev.reference_audio, codec)
        s_clean, s_plain = (r.best() for r in scorer.score([codec.decode(shipped.data, SR), codec.decode(plain.data, SR)], species_idx))
        if s_clean < s_plain - AAC_GUARD_TOLERANCE:
            prev = replace(prev, audio=prev.reference_audio, variant="B", score_preview=prev.score_original,
                           native_preview=prev.native_original,
                           notes=prev.notes + [f"clean-up {prev.variant} dropped after AAC encoding (Perch {s_clean:.2f} vs {s_plain:.2f}); sent the untouched moment"])
            shipped = plain
    prev.timings["encodeS"] = time.perf_counter() - t
    return shipped, prev
