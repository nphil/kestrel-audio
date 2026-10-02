"""The pipeline end to end with a scripted Perch and stand-in separators (real DSP, real loudness, real AAC)."""
import shutil

import numpy as np
import pytest

from kestrel_audio import candidates, loudness
from kestrel_audio.codec import SR, AacCodec
from kestrel_audio.locate import window_starts
from kestrel_audio.pipeline import PreviewError, finalize, make_preview
from kestrel_audio.scoring import WindowScores

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed for the AAC step")


def clip(seconds=12.0, peak=3e-5, seed=0):
    """A far-too-quiet camera clip: hiss with three short 'calls' around 6-7.5 s."""
    rng = np.random.default_rng(seed)
    x = 0.3 * rng.standard_normal(int(seconds * SR))
    t = np.arange(int(0.35 * SR)) / SR
    for start in (6.0, 6.8, 7.4) if seconds > 8 else (1.0, 1.8, 2.4):
        i = int(start * SR)
        x[i:i + len(t)] += 3.0 * np.sin(2 * np.pi * (2500 + 1500 * t) * t) * np.hanning(len(t))
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


def test_unknown_species_ships_the_moment_loud_and_does_no_cleanup():
    prev = make_preview(clip(), species_idx=None, scorer=None, separators=[FakeSeparator(4)])
    assert prev.variant == "B" and not prev.cleaned and prev.method == "trim"
    assert prev.segment.source == "fallback" and any("unknown to Perch" in n for n in prev.notes)
    assert prev.score_original is None
    assert loudness.lufs(prev.audio, SR) == pytest.approx(-16.0, abs=0.5)


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
    assert loudness.lufs(prev.audio, SR) == pytest.approx(-16.0, abs=0.5)
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
    assert loudness.lufs(dec, 48000) == pytest.approx(-16.0, abs=1.0)
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
    for cid in ("M4", "M4S", "M8", "M8S", "MR4_50", "MR8_12"):
        assert candidates.method_of(cid) == "separate"
    for cid in ("M4G", "M4SG", "M8G"):
        assert candidates.method_of(cid) == "separate+gate"
