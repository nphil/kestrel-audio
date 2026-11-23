"""The preview pipeline for one clip: find the matched moment, make it loud, clean it up ONLY if a Perch re-check
proves that helps, then encode. The models arrive as arguments, so the logic can be exercised with fakes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol, Sequence

import numpy as np

from . import dsp, shaping
from .candidates import VOICE, gate_variants, method_of, separated_variants, shape_variants
from .codec import OUT_SR, SR, AacCodec, to_output_rate
from .locate import MIN_EVIDENCE, WIN, Segment, fallback_segment, pick_segment, whole_clip_segment
from .loudness import NormInfo, fade, is_thump, lufs, normalize, pop_gain, prescale, trim_to_true_peak, without_pop
from .scoring import Scorer, Survey
from .speech import SpeechDetector, Voice
from .verify import Candidate, Decision, Reference, choose, select_sources

FADE_MS = 75.0
SILENT_PEAK = 1e-5            # below -100 dBFS a clip is silence, not a quiet animal
POP_KEEPS = 0.9               # a short sound far louder than the rest is not the animal if Perch keeps this share of its confidence without it


class PreviewError(RuntimeError):
    """The clip cannot be turned into a preview (silent, too short, ...)."""


class Separator(Protocol):
    k: int

    def separate(self, x: np.ndarray) -> np.ndarray:
        """x: mono float32 at 22.05 kHz -> (k, len(x)) float32."""
        ...


@dataclass(frozen=True)
class Alternate:
    """A verified candidate held in reserve: what ships if the chosen one does not survive AAC encoding."""
    variant: str
    audio: np.ndarray
    norm: NormInfo
    score: float                      # Perch(species) at preview volume
    native: float                     # ... at the original volume


@dataclass
class Preview:
    audio: np.ndarray                 # what will be shipped (normalised, faded), 22.05 kHz
    reference_audio: np.ndarray       # the untouched moment, normalised: candidate B
    variant: str                      # "B" or a candidate id such as "G20"
    segment: Segment
    norm: NormInfo                    # how `audio` was made loud
    score_original: float | None      # Perch(species) on the untouched moment at preview volume
    score_preview: float | None       # ... on what will be shipped
    native_original: float | None = None
    native_preview: float | None = None
    notes: list[str] = field(default_factory=list)
    decision: list[dict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    survey: Survey | None = None      # Perch's answer for the whole clip, when the caller asked for one (see `surveyor`)
    voice: Voice | None = None        # a human voice heard in the moment
    duck: str | None = None           # how it is turned down in every candidate but B (shaping.DUCK_*); None = it could not be
    call_band: shaping.Band | None = None   # the animal's pitch range as far as Perch let it be cut (None: not asked, or nothing to cut)
    reference_norm: NormInfo | None = None
    alternates: list[Alternate] = field(default_factory=list)

    @property
    def method(self) -> str:
        return method_of(self.variant)

    @property
    def cleaned(self) -> bool:
        return self.variant != "B"

    @property
    def voice_attenuated(self) -> str | None:
        """How the voice in what ships was turned down, None when it was not (or there is none)."""
        return self.duck if (self.voice is not None and self.cleaned) else None


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
                 cleanup: bool = True, surveyor: Callable[[np.ndarray], Survey] | None = None,
                 speech: SpeechDetector | None = None) -> Preview:
    """x22: the whole clip, mono float32 at 22.05 kHz. `surveyor` runs Perch once over the whole clip and keeps every window's
    full answer; the search for the matched moment then reads its curve from that pass instead of making another, and the survey
    comes back on the Preview for whoever wants more than that one species' curve. `speech` listens for human voices in the
    matched moment (it needs the survey)."""
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
    known = species_idx is not None and scorer is not None
    if dur <= WIN:
        seg = whole_clip_segment(dur)
    elif known:
        ws = survey.window_scores(species_idx) if survey is not None else scorer.score([x22], species_idx, full_only=True)[0]
        seg = pick_segment(ws.starts, ws.conf, dur)
        if seg is None:
            seg = fallback_segment(dur)
            notes.append("Perch never heard the species in this clip; used BirdNET-Go's own detection window")
        elif seg.best_conf < MIN_EVIDENCE:
            # a noise-level curve has no real peak, and a score that low is mostly chance: "not lower than before" would be
            # passed or failed by luck, so no clean-up is tried against it
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

    # ---- 2. candidate B: the moment itself, made loud. One short sound far louder than everything else would set the
    # volume of the whole preview; it is turned down first, but only once it is known not to be the animal (a cardinal's
    # chip looks just like a knock): Perch says so where it can, and where it cannot only a low thump is touched
    can_check = known and seg.source == "perch"
    tame_pops = True
    pop = pop_gain(xs, SR)
    if pop is not None:
        t = time.perf_counter()
        if can_check:
            with_it, gone = (r.best() for r in scorer.score([x, without_pop(xs, pop, SR) / scale], species_idx))
            tame_pops = gone >= POP_KEEPS * with_it
            if not tame_pops:
                notes.append("the loudest short sound in this moment is the animal itself (Perch loses it without that sound); it was left as it is")
        else:
            tame_pops = is_thump(xs, pop, SR)
            if not tame_pops:
                notes.append("one short sound in this moment is far louder than the rest and nothing could tell whether it is the animal; "
                             "it was left as it is, so the preview is quieter")
        tm["popS"] = time.perf_counter() - t
    b_final, b_norm = normalize(fade(xs, SR, FADE_MS), SR, tame_pops=tame_pops)
    t = time.perf_counter()
    voice = speech.find(xs, seg.start, survey) if (speech is not None and cleanup) else None
    tm["speechS"] = time.perf_counter() - t
    if voice is not None and not known:
        notes.append("a human voice is heard in this moment; it stays, because Perch does not know the species and could not check what turning it down does to the animal")

    def plain(loud: float | None = None) -> Preview:
        tm["totalS"] = time.perf_counter() - t_all
        return Preview(b_final, b_final, "B", seg, b_norm, loud, loud, notes=notes, timings=tm, survey=survey, voice=voice, reference_norm=b_norm)

    if not can_check and (voice is None or not known):
        return plain()
    if not cleanup:
        t = time.perf_counter()
        loud = scorer.score([b_final], species_idx)[0].best()
        tm["verifyS"] = time.perf_counter() - t
        return plain(loud)

    # ---- 3. AI separation (the GPU/CPU heavy part) and one batched Perch pass over the reference and every track.
    # A moment Perch did not locate gets here only for the voice in it: nothing else is tried on it.
    t = time.perf_counter()
    tracks = {s.k: np.asarray(s.separate(xs), dtype=np.float32) for s in separators} if can_check else {}
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

    # ---- 4. build the candidates. Frames a voice is heard in are left out of every clarity figure.
    t = time.perf_counter()
    scene = shaping.analyse(xs, voice.spans if voice is not None else ())
    keep = scene.free if voice is not None else None
    loud_set, quiet_set = dsp.frame_sets(b_final, keep=keep)[1:]
    ref = Reference(ref_native, ref_loud, dsp.contrast_db(b_final, keep))
    _, base_quiet = dsp.level_in_sets(x, loud_set, quiet_set)
    native_audio: dict[str, np.ndarray] = {}
    loud_audio: dict[str, np.ndarray] = {}
    norms: dict[str, NormInfo] = {}

    def build(cid: str, v: np.ndarray, exempt: bool = False) -> Candidate:
        pre = fade(v, SR, FADE_MS)
        final, info = normalize(pre, SR, tame_pops=tame_pops)
        native = pre / scale
        native_audio[cid], loud_audio[cid], norms[cid] = native, final, info
        return Candidate(cid, dsp.contrast_db(final, keep), base_quiet - dsp.level_in_sets(native, loud_set, quiet_set)[1], info.clean,
                         clicks=len(dsp.new_clicks(b_final, final)), exempt=exempt)

    def native_scores(batch: list[Candidate]) -> dict[str, float]:
        return {c.id: r.best() for c, r in zip(batch, scorer.score([native_audio[c.id] for c in batch], species_idx))}

    def loud_scores(batch: list[Candidate]) -> dict[str, float]:
        return {c.id: r.best() for c, r in zip(batch, scorer.score([loud_audio[c.id] for c in batch], species_idx))}

    def alternate(c: Candidate) -> Alternate:
        return Alternate(c.id, loud_audio[c.id], norms[c.id], c.loud_score, c.native_score)

    def find_band() -> None:
        """Where may the spectrum be cut? Never inside the calls' own range, and outside it as far as Perch still hears the
        animal in its surest window (one batched call, one window per rung)."""
        own = shaping.call_range(scene)
        lows, highs = shaping.rungs_within(own) if own is not None else ([], [])
        if not lows and not highs:
            return
        n_win = int(round(WIN * SR))
        a = int(round((seg.best_start - seg.start) * SR)) if seg.source == "perch" else (len(xs) - n_win) // 2
        a = min(max(a, 0), max(0, len(xs) - n_win))
        probes = [xs] + shaping.band_probes(scene, lows, highs)
        heard = [r.best() for r in scorer.score([p[a:a + n_win] / scale for p in probes], species_idx, full_only=True)]
        fine = [heard[0] > 0.0 and h >= heard[0] * (1.0 - shaping.BAND_SEARCH_LOSS) for h in heard[1:]]
        scene.narrow, scene.wide = shaping.bands_from(lows, highs, fine)

    # ---- 4a. a human voice first: which way of turning it down does Perch accept? That way then goes into every candidate.
    trace: list[dict] = []
    duck: str | None = None
    quiet_voice: Candidate | None = None
    banded = False
    if voice is not None:
        for cid, mode in VOICE.items():            # the whole span first; only if that costs the animal, the band-sparing way
            if mode == shaping.DUCK_OUTSIDE:
                find_band()
                banded = True
            v = shaping.render(scene, duck=mode)
            if v is None:
                continue
            vdec = choose([build(cid, v, exempt=True)], ref, native_scores, loud_scores)
            if vdec.choice is not None:
                quiet_voice, duck = vdec.choice.candidate, mode
            trace += _trace(vdec, ref, None if quiet_voice is None else quiet_voice.id)
            if quiet_voice is not None:
                break
        if duck == shaping.DUCK_OUTSIDE:
            notes.append("a human voice speaks over the animal: it is turned down only outside the animal's own pitch range")
        elif duck is None:
            notes.append("a human voice is heard in this moment; it stays, because turning it down cost Perch some of the animal")
    tm["voiceS"] = time.perf_counter() - t

    # ---- 4b. the clean-ups proper, each with the voice turned down the accepted way
    t = time.perf_counter()
    cands: list[Candidate] = []
    if can_check:
        if not banded:
            find_band()
            banded = True
        variants = gate_variants(xs, dsp.clip_noise_psd(x22 * scale))
        variants.update(shape_variants(scene))
        for k, tr in tracks.items():
            variants.update(separated_variants(k, tr, choices[k]))
        cands = [build(cid, shaping.duck(v, scene, duck) if duck is not None else v) for cid, v in variants.items()]
    tm["buildS"] = time.perf_counter() - t

    # ---- 5. verify with Perch, cleanest first
    t = time.perf_counter()
    dec = choose(cands, ref, native_scores, loud_scores)
    tm["verifyS"] = time.perf_counter() - t
    tm["totalS"] = time.perf_counter() - t_all
    best = dec.choice.candidate if dec.choice is not None else None
    pick = best or quiet_voice
    trace = (trace if best is None else [dict(row, result="verified; a clean-up on top of it was too") if row["result"] == "picked" else row for row in trace])
    trace += _trace(dec, ref, None if best is None else best.id)
    band = scene.narrow if banded else None
    if pick is None:
        return Preview(b_final, b_final, "B", seg, b_norm, ref_loud, ref_loud, native_original=ref_native, native_preview=ref_native,
                       notes=notes, decision=trace, timings=tm, survey=survey, voice=voice, call_band=band, reference_norm=b_norm)
    return Preview(loud_audio[pick.id], b_final, pick.id, seg, norms[pick.id], ref_loud, pick.loud_score, native_original=ref_native,
                   native_preview=pick.native_score, notes=notes, decision=trace, timings=tm, survey=survey, voice=voice, duck=duck,
                   call_band=band, reference_norm=b_norm,
                   alternates=[alternate(quiet_voice)] if (best is not None and quiet_voice is not None) else [])


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
    """Encode the preview. A cleaned preview gets one last Perch check on the DECODED AAC (heavily gated audio can pick up
    codec artefacts): it ships only if Perch is still at least as sure of the species as on the untouched moment, decoded
    the same way. Otherwise the candidate held in reserve is tried, and failing that the untouched moment is shipped."""
    t = time.perf_counter()
    shipped = ship(prev.audio, codec)
    if prev.cleaned and scorer is not None and species_idx is not None:
        untouched = ship(prev.reference_audio, codec)
        s_clean, s_plain = (r.best() for r in scorer.score([codec.decode(shipped.data, SR), codec.decode(untouched.data, SR)], species_idx))
        for alt in ([] if s_clean >= s_plain else prev.alternates + [None]):
            dropped = f"clean-up {prev.variant} dropped after AAC encoding (Perch {s_clean:.3f} vs {s_plain:.3f})"
            if alt is None:
                prev = replace(prev, audio=prev.reference_audio, variant="B", norm=prev.reference_norm or prev.norm, score_preview=prev.score_original,
                               native_preview=prev.native_original, alternates=[], notes=prev.notes + [dropped + "; sent the untouched moment"])
                shipped = untouched
                break
            prev = replace(prev, audio=alt.audio, variant=alt.variant, norm=alt.norm, score_preview=alt.score, native_preview=alt.native,
                           alternates=[], notes=prev.notes + [dropped + f"; sent {alt.variant} instead"])
            shipped = ship(prev.audio, codec)
            s_clean = scorer.score([codec.decode(shipped.data, SR)], species_idx)[0].best()
            if s_clean >= s_plain:
                break
    prev.timings["encodeS"] = time.perf_counter() - t
    return shipped, prev
