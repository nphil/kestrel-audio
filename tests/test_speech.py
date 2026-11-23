"""Human voices: when the voice model's answer becomes a span, and that Perch has to agree before one is reported."""
import numpy as np
import pytest

from kestrel_audio import speech
from kestrel_audio.codec import SR
from kestrel_audio.scoring import Survey
from kestrel_audio.species import Labels

HOP = speech.VAD_CHUNK / speech.VAD_SR          # 32 ms


def probs(n, *runs, level=0.9):
    p = np.zeros(n, dtype=np.float32)
    for a, b in runs:
        p[a:b] = level
    return p


def test_a_run_of_speech_becomes_one_padded_span():
    spans = speech.spans_from(probs(200, (50, 100)))
    assert spans == [pytest.approx((50 * HOP - speech.PAD_BEFORE_S, 100 * HOP + speech.PAD_AFTER_S))]


def test_a_blip_shorter_than_a_word_is_ignored():
    assert speech.spans_from(probs(200, (50, 56))) == []               # 0.19 s
    assert len(speech.spans_from(probs(200, (50, 60)))) == 1           # 0.32 s


def test_the_pause_between_words_belongs_to_the_sentence_a_long_silence_does_not():
    one = speech.spans_from(probs(300, (50, 80), (88, 120)))           # 0.26 s apart
    assert len(one) == 1 and one[0][1] == pytest.approx(120 * HOP + speech.PAD_AFTER_S)
    two = speech.spans_from(probs(300, (50, 80), (130, 160)))          # 1.6 s apart
    assert len(two) == 2 and two[0][1] < two[1][0]


def test_a_voice_lasts_while_the_probability_stays_above_the_lower_threshold():
    p = probs(200, (50, 60))
    p[60:90] = 0.4                                                     # under ON, over OFF: still the same voice
    p[90:100] = 0.2
    assert speech.spans_from(p) == [pytest.approx((50 * HOP - speech.PAD_BEFORE_S, 90 * HOP + speech.PAD_AFTER_S))]
    assert speech.spans_from(np.full(200, 0.4, dtype=np.float32)) == []   # it never started


def test_spans_stay_inside_the_moment():
    spans = speech.spans_from(probs(100, (0, 30), (80, 100)), total_s=3.1)
    assert spans[0][0] == 0.0 and spans[-1][1] == pytest.approx(3.1)


def test_perchs_voice_share_is_taken_over_the_sound_events_only():
    events, voices = np.array([2, 4, 5]), np.array([4])
    logits = np.zeros((2, 6), dtype=np.float32)
    logits[0, 4], logits[1, 2] = 3.0, 3.0
    logits[:, 0] = 9.0                                                 # a loud bird: not an event, so it changes nothing
    share, leads = speech.voice_share(logits, events, voices)
    assert share[0] == pytest.approx(np.exp(3) / (np.exp(3) + 2)) and share[1] == pytest.approx(1 / (np.exp(3) + 2))
    assert list(leads) == [True, False]


def test_voice_classes_are_looked_up_by_name_and_missing_ones_skipped():
    labels = Labels(["Turdus migratorius", "Speech", "Wind", "Laughter", "Chirp_and_tweet"])
    assert list(speech.voice_classes(labels)) == [1, 3]
    assert len(speech.voice_classes(Labels(["Turdus migratorius", "Wind"]))) == 0


def detector(vad_probs):
    return speech.SpeechDetector(lambda x: vad_probs, np.array([4, 5]), np.array([5]))     # classes 4 (wind) and 5 (speech) are the events


def survey(voice_at=(), n=15, logit=3.0):
    """Perch windows every half second: wind in all of them, and a human voice well above it in the ones that start at `voice_at`."""
    starts = np.arange(n) * 0.5
    logits = np.zeros((n, 6), dtype=np.float32)
    logits[:, 4] = 3.0
    for s in voice_at:
        logits[int(round(s / 0.5)), 5] = 3.0 + logit
    return Survey(starts, logits)


def test_a_voice_is_reported_only_when_both_listeners_hear_it():
    xs = np.zeros(6 * SR, dtype=np.float32)
    talking = probs(187, (40, 80))                                     # 1.28 to 2.56 s into the moment
    quiet_perch = survey()                                             # nothing but wind
    both = detector(talking).find(xs, 4.0, survey(voice_at=[3.0]))    # the moment starts 4 s into the clip: the voice is at 5.1 to 6.8 s
    assert both is not None and both.share == pytest.approx(np.exp(3) / (np.exp(3) + 1))
    assert both.spans == (pytest.approx((40 * HOP - speech.PAD_BEFORE_S, 80 * HOP + speech.PAD_AFTER_S)),)
    assert both.seconds == pytest.approx(40 * HOP + speech.PAD_BEFORE_S + speech.PAD_AFTER_S)
    assert detector(talking).find(xs, 4.0, quiet_perch) is None                                   # the voice model alone: a bird that sounds like speech
    assert detector(probs(187)).find(xs, 4.0, survey(voice_at=[3.0])) is None                     # Perch alone: an owl it calls "singing"
    assert detector(talking).find(xs, 4.0, survey(voice_at=[0.0])) is None                        # Perch's voice is in another part of the clip (0 to 5 s)
    assert detector(talking).find(xs, 4.0, None) is None
    assert speech.SpeechDetector(lambda x: talking, np.array([4, 5]), np.array([], dtype=np.int64)).find(xs, 4.0, survey(voice_at=[3.0])) is None


def test_a_leading_voice_class_is_enough_even_at_a_small_share():
    xs = np.zeros(6 * SR, dtype=np.float32)
    many = speech.SpeechDetector(lambda x: probs(187, (40, 80)), np.arange(4, 24), np.array([5]))  # twenty event classes, one of them a voice
    logits = np.zeros((15, 24), dtype=np.float32)
    logits[6, 5] = 0.5                                                 # barely ahead of nineteen others: 8% of the belief
    found = many.find(xs, 4.0, Survey(np.arange(15) * 0.5, logits))
    assert found is not None and found.share < speech.MIN_SHARE
