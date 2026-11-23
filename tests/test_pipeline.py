"""The pipeline end to end with a scripted Perch and stand-in separators (real DSP, real loudness, real AAC)."""
import shutil

import numpy as np
import pytest

from kestrel_audio import candidates, dsp, loudness, speech
from kestrel_audio.codec import SR, AacCodec
from kestrel_audio.locate import window_starts
from kestrel_audio.pipeline import PreviewError, finalize, make_preview
from kestrel_audio.scoring import Survey, WindowScores

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed for the AAC step")


VOICE_HZ = 200.0


def talk(seconds):
    """A stand-in voice: a 200 Hz buzz with overtones up to 2.4 kHz, in syllables."""
    t = np.arange(int(seconds * SR)) / SR
    buzz = sum(np.sin(2 * np.pi * VOICE_HZ * k * t) / k for k in range(1, 13))
    return 1.5 * buzz * (0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)) * np.hanning(len(t)) ** 0.25


def clip(seconds=12.0, peak=3e-5, seed=0, voice=None, call=(0.35, 3.0)):
    """A far-too-quiet camera clip: hiss with three short 'calls' around 6-7.5 s, each `call` = (seconds long, times as high as
    the hiss is strong / 0.3). `voice` = (start, seconds) has someone talk in it."""
    rng = np.random.default_rng(seed)
    x = 0.3 * rng.standard_normal(int(seconds * SR))
    t = np.arange(int(call[0] * SR)) / SR
    for start in (6.0, 6.8, 7.4) if seconds > 8 else (1.0, 1.8, 2.4):
        i = int(start * SR)
        x[i:i + len(t)] += call[1] * np.sin(2 * np.pi * (2500 + 1500 * t) * t) * np.hanning(len(t))
    if voice is not None:
        v = talk(voice[1])
        i = int(voice[0] * SR)
        x[i:i + len(v)] += v
    return (x / np.max(np.abs(x)) * peak).astype(np.float32)


class FakeScorer:
    """Perch stand-in. `conf(array, call_no, full_only)` returns the per-window confidence; default = `base` everywhere,
    with a peak on the windows that cover 6-7.5 s when locating."""

    def __init__(self, conf=None, base=0.8):
        self.calls = []
        self._conf = conf
        self.base = base

    def score(self, arrays, species_idx, *, full_only=False):
        self.calls.append((len(arrays), full_only))
        out = []
        for a in arrays:
            starts = window_starts(len(a) / SR, full_only=full_only) or [0.0]
            if self._conf:
                conf = np.array([self._conf(a, len(self.calls), full_only, s) for s in starts])
            else:
                conf = np.full(len(starts), self.base)
            out.append(WindowScores(np.array(starts), conf, np.maximum(conf, 0.1)))
        return out


class FakeSeparator:
    def __init__(self, k):
        self.k = k

    def separate(self, x):
        rng = np.random.default_rng(self.k)
        return np.stack([x * 0.7 + 0.02 * rng.standard_normal(len(x)).astype(np.float32) * np.max(np.abs(x))] +
                        [x * 0.1 * (j + 1) for j in range(self.k - 1)]).astype(np.float32)


def locate_peak(a, call_no, full_only, start):
    """Perch is sure of the animal only in windows that overlap 6-7.5 s of the clip (first call = the locating pass)."""
    return 0.9 if (call_no == 1 and start <= 6.0 and start + 5.0 >= 7.5) else (0.05 if call_no == 1 else 0.8)


def as_loud_as_the_hiss_allows(prev):
    """At the target loudness, or under it by exactly what was given up so the hiss stays in the background (6 dB at most)."""
    cut = prev.norm.floor_cut_db
    assert 0.0 <= cut <= loudness.FLOOR_MAX_CUT_DB
    assert loudness.lufs(prev.audio, SR) == pytest.approx(-16.0 - cut, abs=0.5)


def test_unknown_species_ships_the_moment_loud_and_does_no_cleanup():
    prev = make_preview(clip(), species_idx=None, scorer=None, separators=[FakeSeparator(4)])
    assert prev.variant == "B" and not prev.cleaned and prev.method == "trim"
    assert prev.segment.source == "fallback" and any("unknown to Perch" in n for n in prev.notes)
    assert prev.score_original is None
    as_loud_as_the_hiss_allows(prev)


def test_silent_or_tiny_clips_are_rejected_not_amplified():
    with pytest.raises(PreviewError, match="silent"):
        make_preview(np.zeros(10 * SR, dtype=np.float32), species_idx=None, scorer=None, separators=[])
    with pytest.raises(PreviewError, match="half a second"):
        make_preview(np.ones(1000, dtype=np.float32) * 0.1, species_idx=None, scorer=None, separators=[])


def test_a_clip_no_longer_than_one_window_is_used_whole_and_not_scored():
    scorer = FakeScorer()
    prev = make_preview(clip(seconds=4.0), species_idx=3, scorer=scorer, separators=[FakeSeparator(4)])
    assert prev.segment.source == "whole" and prev.variant == "B" and scorer.calls == []


def test_the_matched_moment_is_found_and_a_quiet_clip_comes_out_at_the_target_loudness():
    # call 1 locates, call 2 scores the untouched moment and the AI tracks (all sure), later calls re-check candidates (all lost)
    scorer = FakeScorer(conf=lambda a, n, f, s: locate_peak(a, n, f, s) if n <= 2 else 0.0)
    prev = make_preview(clip(), species_idx=3, scorer=scorer, separators=[FakeSeparator(4), FakeSeparator(8)])
    seg = prev.segment
    assert seg.source == "perch" and seg.start < 6.0 and seg.end > 7.5          # holds the calls
    assert seg.length < 12.0
    assert prev.variant == "B" and not prev.cleaned                            # no candidate re-checked as good as the original
    as_loud_as_the_hiss_allows(prev)
    assert len(prev.decision) >= 10 and all(d["result"] for d in prev.decision)
    assert prev.timings["locateS"] >= 0 and prev.timings["totalS"] > 0


def test_a_candidate_that_perch_still_recognises_replaces_the_untouched_moment():
    scorer = FakeScorer(conf=locate_peak)                                      # every re-check scores 0.8, same as the original
    prev = make_preview(clip(), species_idx=3, scorer=scorer, separators=[FakeSeparator(4)])
    assert prev.cleaned and prev.variant != "B"
    assert prev.method == candidates.method_of(prev.variant)
    picked = [d for d in prev.decision if d["result"] == "picked"]
    assert len(picked) == 1 and picked[0]["id"] == prev.variant and picked[0]["clarityGainDb"] >= 3.0
    assert prev.score_preview == pytest.approx(0.8) and prev.score_original == pytest.approx(0.8)
    best = max(prev.decision, key=lambda d: d["clarityGainDb"] if d["perchLoud"] is not None else -1)
    assert best["id"] == prev.variant                                        # the clearest verified candidate won


def test_without_cleanup_only_the_untouched_moment_is_scored():
    scorer = FakeScorer(conf=locate_peak)
    prev = make_preview(clip(), species_idx=3, scorer=scorer, separators=[FakeSeparator(4)], cleanup=False)
    assert prev.variant == "B" and prev.score_original == pytest.approx(0.8)
    assert [n for n, _ in scorer.calls] == [1, 1]                              # locate, then the one score of B


@needs_ffmpeg
def test_shipped_file_meets_loudness_and_true_peak_after_the_aac_round_trip():
    scorer = FakeScorer(conf=locate_peak)
    codec = AacCodec()
    prev = make_preview(clip(), species_idx=3, scorer=scorer, separators=[FakeSeparator(4)])
    shipped, prev = finalize(prev, codec=codec, scorer=scorer, species_idx=3)
    dec = codec.decode(shipped.data)
    assert shipped.data[4:8] == b"ftyp"                                        # an MP4/M4A container
    assert loudness.true_peak_db(dec) <= -1.0 + 0.05                           # measured on the DECODED file
    assert loudness.lufs(dec, 48000) == pytest.approx(-16.0 - prev.norm.held_db, abs=1.0)
    assert shipped.duration_s == pytest.approx(prev.segment.length, abs=0.1)


@needs_ffmpeg
def test_a_clean_up_that_loses_the_animal_in_the_aac_step_is_replaced_by_the_untouched_moment():
    prev = make_preview(clip(), species_idx=3, scorer=FakeScorer(conf=locate_peak), separators=[FakeSeparator(4)])
    assert prev.cleaned
    cleaned_id = prev.variant

    class Guard:                                   # Perch scoring the decoded AAC: the clean-up falls apart, the plain moment does not
        def score(self, arrays, species_idx, *, full_only=False):
            return [WindowScores(np.array([0.0]), np.array([0.1]), np.array([0.1])),
                    WindowScores(np.array([0.0]), np.array([0.8]), np.array([0.8]))]

    shipped, final = finalize(prev, codec=AacCodec(), scorer=Guard(), species_idx=3)
    assert final.variant == "B" and not final.cleaned and final.method == "trim"
    assert any(cleaned_id in n and "AAC" in n for n in final.notes)


def test_every_candidate_id_maps_to_one_of_the_documented_methods():
    assert candidates.method_of("B") == "trim"
    assert {candidates.method_of(c) for c in candidates.GATES} == {"gate"}
    assert {candidates.method_of(c) for c in candidates.SHAPES} == {"shape"}
    assert {candidates.method_of(c) for c in candidates.VOICE} == {"voice"}
    for cid in ("M4", "M4S", "M8", "M8S", "MR4_50", "MR8_12"):
        assert candidates.method_of(cid) == "separate"
    for cid in ("M4G", "M4SG", "M8G"):
        assert candidates.method_of(cid) == "separate+gate"


def test_a_noise_level_perch_curve_is_not_a_match_and_nothing_is_cleaned():
    # Perch's best window is 0.08: there is no real peak to find, and "not lower than on the untouched moment" would be passed
    # or failed by luck, so the moment is BirdNET-Go's own window and the separators never run
    scorer = FakeScorer(base=0.08)
    prev = make_preview(clip(), species_idx=3, scorer=scorer, separators=[FakeSeparator(4), FakeSeparator(8)])
    assert prev.segment.source == "fallback" and prev.segment.start == pytest.approx(2.5)
    assert prev.variant == "B" and not prev.cleaned and prev.decision == []
    assert any("barely sure" in n and "0.08" in n for n in prev.notes)
    assert scorer.calls == [(1, True)]                                          # located, then nothing else was scored


def test_a_curve_just_at_the_evidence_floor_still_counts_as_a_match():
    scorer = FakeScorer(base=0.10)
    prev = make_preview(clip(), species_idx=3, scorer=scorer, separators=[])
    assert prev.segment.source == "perch"


def test_one_survey_of_the_whole_clip_replaces_the_locating_pass_and_comes_back_on_the_preview():
    c = clip()
    starts = np.array(window_starts(len(c) / SR, full_only=True))
    logits = np.zeros((len(starts), 6), dtype=np.float32)
    logits[(starts >= 2.5) & (starts <= 6.0), 3] = 6.0              # class 3 stands out in the windows that cover 6-7.5 s
    survey, asked = Survey(starts, logits), []
    scorer = FakeScorer(base=0.8)
    prev = make_preview(c, species_idx=3, scorer=scorer, separators=[], cleanup=False,
                        surveyor=lambda x: asked.append(len(x)) or survey)
    assert asked == [len(c)] and prev.survey is survey
    assert prev.segment.source == "perch" and prev.segment.start < 6.0 and prev.segment.end > 7.5      # located from the survey's curve
    assert scorer.calls == [(1, False)]                                                                 # only the re-score of the moment: no second pass over the clip


def test_without_a_surveyor_nothing_is_surveyed_and_the_locating_pass_is_the_scorers_own():
    scorer = FakeScorer(conf=locate_peak)
    prev = make_preview(clip(), species_idx=3, scorer=scorer, separators=[], cleanup=False)
    assert prev.survey is None and scorer.calls[0] == (1, True)


def test_a_clip_with_no_known_species_or_no_window_to_search_can_still_be_surveyed():
    sentinel = Survey(np.array([0.0]), np.zeros((1, 3), dtype=np.float32))
    unknown = make_preview(clip(), species_idx=None, scorer=None, separators=[], surveyor=lambda x: sentinel)
    short = make_preview(clip(seconds=4.0), species_idx=3, scorer=FakeScorer(), separators=[], surveyor=lambda x: sentinel)
    assert unknown.survey is sentinel and unknown.segment.source == "fallback"
    assert short.survey is sentinel and short.segment.source == "whole"


# ------------------------------------------------------------------------------------------------ clicks where the moment was cut

def hum(seconds=12.0, peak=3e-5):
    """A steady hum right through the clip: wherever the moment is cut out of it, the cut lands on the top of a wave."""
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(4)
    return (peak * (np.cos(2 * np.pi * 233.0 * t) + 0.003 * rng.standard_normal(len(t)))).astype(np.float32)


def edges(y):
    """The largest sample in the first and in the last millisecond, against the largest anywhere."""
    n = int(1e-3 * SR)
    return float(max(np.abs(y[:n]).max(), np.abs(y[-n:]).max()) / np.abs(y).max())


def test_the_preview_does_not_click_where_it_was_cut_out_of_the_clip():
    c = hum()
    prev = make_preview(c, species_idx=3, scorer=FakeScorer(conf=locate_peak), separators=[], cleanup=False)
    seg = prev.segment
    assert seg.start > 0.0 and seg.end < 12.0                                    # cut at both ends
    bare = c[int(round(seg.start * SR)):int(round(seg.end * SR))]
    bare = bare / np.abs(bare).max()
    clicks = dsp.click_times(bare)
    assert edges(bare) > 0.5 and len(clicks) == 1 and clicks[0] < 0.002          # what the bare cut is: full scale from the first sample
    assert edges(prev.audio) < 0.01 and len(dsp.click_times(prev.audio)) == 0


@needs_ffmpeg
def test_the_shipped_file_starts_and_ends_in_silence():
    scorer = FakeScorer(conf=locate_peak)
    codec = AacCodec()
    shipped, _ = finalize(make_preview(hum(), species_idx=3, scorer=scorer, separators=[], cleanup=False), codec=codec, scorer=scorer, species_idx=3)
    dec = codec.decode(shipped.data, SR)
    assert edges(dec) < 0.02 and len(dsp.click_times(dec)) == 0


def test_no_clean_up_that_ships_has_a_click_the_untouched_moment_does_not():
    prev = make_preview(clip(), species_idx=3, scorer=FakeScorer(conf=locate_peak), separators=[FakeSeparator(4)])
    assert prev.cleaned and len(dsp.new_clicks(prev.reference_audio, prev.audio)) == 0
    assert edges(prev.audio) < 0.01


# ------------------------------------------------------------------------------------------------ human voices

def hears_the_buzz(x):
    """Stands in for the voice model: 'speech' in every 32 ms in which the 200 Hz buzz is strong."""
    hop = int(round(0.032 * SR))
    n = len(x) // hop
    ref = np.exp(-2j * np.pi * VOICE_HZ * np.arange(hop) / SR)
    mag = np.abs(np.asarray(x[:n * hop], dtype=np.float64).reshape(n, hop) @ ref) / hop
    return (mag > 0.03 * np.abs(x).max()).astype(np.float32)


def hears_the_chirps(a, call_no, full_only, start):
    """Stands in for Perch, and listens: it is as sure of the animal as the share of the window in which its chirps (2.5 to 3.6 kHz)
    stand clear of what is usually higher up. Take a chirp away and it is less sure; hiss, hum, voices and a knock do not impress it."""
    w = np.asarray(a[int(start * SR):int((start + 5.0) * SR)], dtype=np.float32)
    if len(w) < SR:
        return 0.0
    f, _, Z = dsp._stft(w)
    P = np.abs(Z) ** 2
    inside, above = dsp.band_ms_db(P, f, 2400.0, 3800.0), dsp.band_ms_db(P, f, 5000.0, 9000.0)
    return 0.8 * min(1.0, float(np.mean(inside >= np.percentile(above, 90) + 10.0)) / 0.12)


LISTENER = speech.SpeechDetector(hears_the_buzz, np.array([5, 6]), np.array([6]))     # of Perch's classes, 5 (wind) and 6 (speech) are sound events


def heard(c, voice_in, animal=6.0, elsewhere=0.0):
    """Perch's answer for the whole clip: the animal (class 3) in the windows that cover 6-7.5 s (`elsewhere` in the others), wind in
    all of them, and a human voice in the windows `voice_in` picks."""
    starts = np.array(window_starts(len(c) / SR, full_only=True))
    logits = np.zeros((len(starts), 8), dtype=np.float32)
    logits[:, 3] = elsewhere
    logits[(starts >= 2.5) & (starts <= 6.0), 3] = animal
    logits[:, 5] = 1.0
    logits[voice_in(starts), 6] = 3.0
    return Survey(starts, logits)


def buzz_against_chirps(y, a, b):
    """How loud the voice's 200 Hz buzz is between `a` and `b` seconds, against the loudest the chirps get (dB)."""
    part = np.asarray(y[int(a * SR):int(b * SR)], dtype=np.float64)
    buzz = np.abs(part @ np.exp(-2j * np.pi * VOICE_HZ * np.arange(len(part)) / SR)) / len(part)
    f, _, Z = dsp._stft(np.asarray(y, dtype=np.float32))
    return float(20.0 * np.log10(buzz + 1e-12) - np.percentile(dsp.band_ms_db(np.abs(Z) ** 2, f, 2400.0, 3800.0), 99))


def test_a_voice_beside_the_calls_is_turned_down_and_the_calls_are_not():
    c = clip(voice=(8.6, 1.0))
    survey = heard(c, lambda s: s >= 4.0)
    prev = make_preview(c, species_idx=3, scorer=FakeScorer(conf=hears_the_chirps), separators=[], surveyor=lambda x: survey, speech=LISTENER)
    seg = prev.segment
    assert prev.voice is not None and len(prev.voice.spans) == 1 and prev.voice.seconds == pytest.approx(1.4, abs=0.3)
    assert prev.voice.spans[0][0] + seg.start == pytest.approx(8.45, abs=0.15) and prev.voice.spans[0][1] + seg.start == pytest.approx(9.85, abs=0.15)
    assert prev.voice_attenuated == "full" and prev.cleaned
    a, b = 8.8 - seg.start, 9.4 - seg.start
    assert buzz_against_chirps(prev.audio, a, b) < buzz_against_chirps(prev.reference_audio, a, b) - 30.0
    assert prev.score_preview >= prev.score_original and prev.native_preview >= prev.native_original       # all three chirps still heard
    first = prev.decision[0]
    assert first["id"] == "V2" and "verified" in first["result"]                # the voice first, then a clean-up on top of it


def test_a_voice_over_a_call_is_turned_down_only_outside_the_animals_pitch_range():
    c = clip(voice=(7.2, 1.4))                                                    # it talks over the third chirp
    survey = heard(c, lambda s: s >= 2.5)
    prev = make_preview(c, species_idx=3, scorer=FakeScorer(conf=hears_the_chirps), separators=[], surveyor=lambda x: survey, speech=LISTENER)
    seg = prev.segment
    whole = next(d for d in prev.decision if d["id"] == "V2")
    assert "original volume" in whole["result"]                                  # turning the whole stretch down took a chirp with it
    assert prev.voice_attenuated == "outside" and any("speaks over the animal" in n for n in prev.notes)
    assert prev.call_band is not None and prev.call_band[0] >= 900.0             # the animal's range starts above the body of the voice
    a, b = 7.5 - seg.start, 8.3 - seg.start
    assert buzz_against_chirps(prev.audio, a, b) < buzz_against_chirps(prev.reference_audio, a, b) - 25.0
    assert prev.native_preview >= prev.native_original                          # the third chirp is still there


def test_a_voice_that_cannot_be_told_from_the_calls_stays_and_the_preview_says_so():
    c = clip(voice=(5.8, 2.2))                                                    # it talks over every chirp: nothing says where the animal's own range is
    survey = heard(c, lambda s: s >= 1.5)
    prev = make_preview(c, species_idx=3, scorer=FakeScorer(conf=hears_the_chirps), separators=[], surveyor=lambda x: survey, speech=LISTENER)
    assert prev.voice is not None and prev.voice_attenuated is None
    assert any("it stays" in n for n in prev.notes)
    a, b = 6.3 - prev.segment.start, 7.6 - prev.segment.start
    assert buzz_against_chirps(prev.audio, a, b) > buzz_against_chirps(prev.reference_audio, a, b) - 6.0


def test_on_a_moment_perch_could_not_locate_only_the_voice_is_touched():
    c = clip(voice=(3.6, 1.0))
    survey = heard(c, lambda s: s <= 4.5, animal=-3.0, elsewhere=-6.0)           # Perch is barely sure of the animal anywhere
    scorer = FakeScorer(conf=hears_the_chirps)
    prev = make_preview(c, species_idx=3, scorer=scorer, separators=[FakeSeparator(4)], surveyor=lambda x: survey, speech=LISTENER)
    assert prev.segment.source == "fallback" and any("barely sure" in n for n in prev.notes)
    assert prev.variant == "V2" and prev.method == "voice" and prev.voice_attenuated == "full"
    assert [d["id"] for d in prev.decision] == ["V2"] and prev.decision[0]["result"] == "picked"
    assert prev.score_preview >= prev.score_original


def test_a_voice_stays_when_perch_does_not_know_the_species():
    c = clip(voice=(3.6, 1.0))
    survey = heard(c, lambda s: s <= 4.5)
    prev = make_preview(c, species_idx=None, scorer=None, separators=[], surveyor=lambda x: survey, speech=LISTENER)
    assert prev.variant == "B" and prev.voice is not None and prev.voice_attenuated is None
    assert any("Perch does not know the species" in n for n in prev.notes)


def test_without_a_voice_model_or_without_a_voice_nothing_about_voices_is_reported():
    c = clip(voice=(8.6, 1.0))
    survey = heard(c, lambda s: s >= 4.0)
    deaf = make_preview(c, species_idx=3, scorer=FakeScorer(conf=hears_the_chirps), separators=[], surveyor=lambda x: survey)
    assert deaf.voice is None and deaf.voice_attenuated is None
    plain = clip()
    none = make_preview(plain, species_idx=3, scorer=FakeScorer(conf=hears_the_chirps), separators=[], surveyor=lambda x: heard(plain, lambda s: s < 0), speech=LISTENER)
    assert none.voice is None and not any("voice" in n for n in none.notes)


@needs_ffmpeg
def test_a_clean_up_lost_in_the_aac_step_falls_back_to_the_voice_only_version_before_the_untouched_moment():
    c = clip(voice=(8.6, 1.0))
    survey = heard(c, lambda s: s >= 4.0)
    prev = make_preview(c, species_idx=3, scorer=FakeScorer(conf=hears_the_chirps), separators=[], surveyor=lambda x: survey, speech=LISTENER)
    assert prev.cleaned and prev.variant != "V2" and [a.variant for a in prev.alternates] == ["V2"]

    class Guard:                                   # Perch on the decoded AAC: the clean-up falls apart, the voice-only version holds
        def __init__(self):
            self.answers = [[0.1, 0.8], [0.8]]

        def score(self, arrays, species_idx, *, full_only=False):
            return [WindowScores(np.array([0.0]), np.array([v]), np.array([v])) for v in self.answers.pop(0)]

    shipped, final = finalize(prev, codec=AacCodec(), scorer=Guard(), species_idx=3)
    assert final.variant == "V2" and final.voice_attenuated == "full"
    assert any(prev.variant in n and "sent V2 instead" in n for n in final.notes)


# ------------------------------------------------------------------------------------------------ one short sound far louder than the rest

KNOCK = np.random.default_rng(9).standard_normal(int(0.0015 * SR)) * np.hanning(int(0.0015 * SR))      # 1.5 ms, white: in every band
THUMP = np.sin(np.pi * np.arange(SR // 80) / (SR // 80))                                                 # half a 40 Hz wave: 12 ms
CHIP = np.sin(2 * np.pi * 7000.0 * np.arange(int(0.008 * SR)) / SR) * np.hanning(int(0.008 * SR))        # 8 ms at 7 kHz


def louder(c, at, shape, times=5.0):
    """`c` with `shape` added at `at` seconds, `times` as high as anything else in the clip."""
    y = np.array(c, dtype=np.float32)
    i = int(at * SR)
    y[i:i + len(shape)] += (times * np.abs(c).max() * shape / np.abs(shape).max()).astype(np.float32)
    return y


def crest_db(y):
    """How far the highest sample stands above the usual one."""
    m = np.abs(np.asarray(y, dtype=np.float64))
    return float(20.0 * np.log10(m.max() / np.median(m)))


def test_a_knock_is_turned_down_so_it_does_not_set_the_volume_once_perch_says_it_is_not_the_animal():
    plain = clip()
    c = louder(plain, 6.5, KNOCK)                                                 # between the first two chirps, 14 dB above them
    survey = heard(c, lambda s: s < 0)
    scorer = FakeScorer(conf=hears_the_chirps)
    prev = make_preview(c, species_idx=3, scorer=scorer, separators=[], surveyor=lambda x: survey, cleanup=False)
    assert scorer.calls == [(2, False), (1, False)]                             # the moment with and without the knock, then the usual one score
    assert prev.norm.tamed_db > 6.0 and not any("left as it is" in n for n in prev.notes)
    same = make_preview(plain, species_idx=3, scorer=FakeScorer(conf=hears_the_chirps), separators=[], surveyor=lambda x: survey, cleanup=False)
    assert same.norm.tamed_db == 0.0
    assert loudness.lufs(prev.audio, SR) == pytest.approx(loudness.lufs(same.audio, SR), abs=1.0)      # as loud as without the knock


def test_a_chip_that_is_the_animal_is_left_as_it_is_however_much_louder_than_the_rest():
    c = clip(call=(0.015, 6.0))                                                   # the whole call is three 15 ms chips: shorter than any knock
    survey = heard(c, lambda s: s < 0)
    prev = make_preview(c, species_idx=3, scorer=FakeScorer(conf=hears_the_chirps), separators=[], surveyor=lambda x: survey, cleanup=False)
    seg = prev.segment
    raw = c[int(round(seg.start * SR)):int(round(seg.end * SR))]
    assert any("is the animal itself" in n for n in prev.notes) and prev.norm.tamed_db == 0.0
    assert crest_db(prev.audio) > crest_db(raw) - 2.0                             # the chips stand as high above the hiss as they did
    flattened, info = loudness.normalize(loudness.prescale(raw)[0], SR)           # what turning them down would have done
    assert info.tamed_db > 6.0 and crest_db(flattened) < crest_db(raw) - 6.0


def test_where_perch_cannot_tell_only_a_low_thump_is_turned_down():
    thump = make_preview(louder(clip(), 5.0, THUMP), species_idx=None, scorer=None, separators=[])
    assert thump.norm.tamed_db > 6.0 and not any("left as it is" in n for n in thump.notes)
    as_loud_as_the_hiss_allows(thump)
    chip = make_preview(louder(clip(), 5.0, CHIP), species_idx=None, scorer=None, separators=[])        # it could be a bird: hands off
    assert chip.norm.tamed_db == 0.0 and any("left as it is" in n for n in chip.notes)
    assert chip.norm.held_db > 6.0 and chip.norm.max_gr_db <= loudness.CALL_LIMIT_DB + 0.5              # quieter, and not squashed
