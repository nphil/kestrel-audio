"""The turn-down-only clean-ups: where the calls are, what may be cut away, and that the calls themselves are left alone."""
import numpy as np
import pytest

from kestrel_audio import dsp, shaping

SR = dsp.SR
CALLS = ((1.0, 0.4), (2.7, 0.4), (4.2, 0.4))


def moment(seconds=6.0, hiss=0.02, calls=CALLS, tone=3000.0, seed=0):
    """Hiss with a few tone bursts: the animal calls at `tone` Hz, and only there."""
    rng = np.random.default_rng(seed)
    x = hiss * rng.standard_normal(int(seconds * SR))
    for start, length in calls:
        t = np.arange(int(length * SR)) / SR
        i = int(start * SR)
        x[i:i + len(t)] += 0.5 * np.sin(2 * np.pi * tone * t) * np.hanning(len(t))
    return x.astype(np.float32)


def level(x, a, b, lo=dsp.LOW_HZ, hi=dsp.HIGH_HZ):
    """Mean level (dB) between `a` and `b` seconds, in the band lo..hi."""
    f, _, Z = dsp._stft(np.asarray(x[int(a * SR):int(b * SR)], dtype=np.float32))
    return float(10.0 * np.log10(np.mean(10.0 ** (dsp.band_ms_db(np.abs(Z) ** 2, f, lo, hi) / 10.0))))


def test_the_calls_own_range_is_read_from_the_loud_frames_minus_the_quiet_ones():
    lo, hi = shaping.call_range(shaping.analyse(moment()))
    assert 2300.0 < lo < 2520.0 and 3580.0 < hi < 3800.0             # 3 kHz, a fifth clear on each side
    lows, highs = shaping.rungs_within((lo, hi))
    assert lows == [500.0, 900.0, 1500.0, 2400.0] and highs == [8500.0, 6500.0, 5000.0, 3800.0]
    assert shaping.rungs_within((700.0, 6000.0)) == ([500.0], [8500.0, 6500.0])
    assert shaping.rungs_within((200.0, 9000.0)) == ([], [])          # an owl below, a hiss-like call above: nothing may be cut


def test_a_steady_sound_has_no_range_to_read_and_nothing_is_cut():
    assert shaping.call_range(shaping.analyse(moment(calls=()))) is None
    steady = moment(calls=()) + (0.3 * np.sin(2 * np.pi * 3000 * np.arange(6 * SR) / SR)).astype(np.float32)
    assert shaping.call_range(shaping.analyse(steady)) is None        # a trill from end to end: no quiet frames to compare with


def test_the_band_stops_at_the_first_rung_perch_objects_to():
    lows, highs = [500.0, 900.0, 1500.0], [8500.0, 6500.0]
    narrow, wide = shaping.bands_from(lows, highs, [True, True, False, True, True])
    assert narrow == (900.0, 6500.0) and wide == (500.0, 8500.0)
    assert shaping.bands_from(lows, highs, [False, True, True, False, True]) == (None, None)       # a later rung does not count without the earlier one
    assert shaping.bands_from(lows, highs, [True, False, False, False, False]) == ((500.0, float("inf")), None)
    assert shaping.bands_from([], highs, [True, True]) == ((0.0, 6500.0), (0.0, 8500.0))


def test_presence_is_one_on_the_calls_and_zero_between_them():
    scene = shaping.analyse(moment())
    p = shaping.presence(scene, None)
    t = np.arange(len(p)) * dsp.HOP / SR
    for start, length in CALLS:
        assert p[(t > start + 0.08) & (t < start + length - 0.08)].min() == pytest.approx(1.0)
    assert p[(t > 2.1) & (t < 2.55)].max() < 0.02 and p[t > 5.4].max() < 0.02
    assert np.all(shaping.presence(shaping.analyse(moment(calls=())), None) == 1.0)                 # nothing stands out: nothing to hush between


def test_the_hush_lowers_only_the_stretches_between_calls():
    x = moment()
    y = shaping.render(shaping.analyse(x), hush_db=14.0)
    assert len(y) == len(x)
    assert level(y, 1.9, 2.5) == pytest.approx(level(x, 1.9, 2.5) - 14.0, abs=1.0)
    for start, length in CALLS:
        a, b = int((start + 0.08) * SR), int((start + length - 0.08) * SR)
        assert np.max(np.abs(y[a:b] - x[a:b])) < 1e-3 * np.max(np.abs(x))                           # the call itself: sample for sample


def test_a_band_cut_leaves_the_call_and_removes_what_lies_outside():
    x = moment()
    y = shaping.render(shaping.analyse(x), band=(1500.0, 5000.0))
    assert level(y, 1.0, 1.4, 2500.0, 3500.0) == pytest.approx(level(x, 1.0, 1.4, 2500.0, 3500.0), abs=0.2)
    assert level(y, 0.0, 6.0, 7000.0, 10000.0) < level(x, 0.0, 6.0, 7000.0, 10000.0) - 25.0
    assert level(y, 0.0, 6.0, 300.0, 1100.0) < level(x, 0.0, 6.0, 300.0, 1100.0) - 25.0
    g = shaping.band_gain(np.array([100.0, 1200.0, 1500.0, 3000.0, 5000.0, 6250.0, 9000.0]), (1500.0, 5000.0))
    assert np.allclose(g[[2, 3, 4]], 1.0) and np.allclose(g[[0, 1, 5, 6]], 10 ** (shaping.BAND_DEPTH_DB / 20.0), atol=1e-6)


def test_a_voice_span_is_turned_down_whole_or_only_outside_the_animals_band():
    x = moment()
    scene = shaping.analyse(x, [(1.9, 2.5)])
    assert scene.voiced.any() and not scene.voiced.all() and int(scene.free.sum()) == int((~scene.voiced).sum())
    whole = shaping.render(scene, duck=shaping.DUCK_FULL)
    assert level(whole, 2.05, 2.35) < level(x, 2.05, 2.35) - 35.0
    assert level(whole, 1.0, 1.4) == pytest.approx(level(x, 1.0, 1.4), abs=0.2) and level(whole, 2.7, 3.1) == pytest.approx(level(x, 2.7, 3.1), abs=0.2)
    assert shaping.render(scene, duck=shaping.DUCK_OUTSIDE) is None                                # no band granted: that way is not on offer
    scene.narrow = (500.0, 5000.0)
    assert shaping.render(scene, duck=shaping.DUCK_OUTSIDE) is None                                # nor with a band that holds the body of the voice
    scene.narrow = (1500.0, 5000.0)
    spared = shaping.render(scene, duck=shaping.DUCK_OUTSIDE)
    assert level(spared, 2.05, 2.35, 2000.0, 4000.0) == pytest.approx(level(x, 2.05, 2.35, 2000.0, 4000.0), abs=0.5)
    assert level(spared, 2.05, 2.35, 7000.0, 10000.0) < level(x, 2.05, 2.35, 7000.0, 10000.0) - 25.0
    assert level(spared, 2.05, 2.35, 300.0, 1100.0) < level(x, 2.05, 2.35, 300.0, 1100.0) - 25.0
    other = shaping.duck((0.5 * x).astype(np.float32), scene, shaping.DUCK_FULL)                   # any other candidate, ducked the same way
    assert level(other, 2.05, 2.35) < level(x, 2.05, 2.35) - 35.0 and level(other, 1.0, 1.4) == pytest.approx(level(x, 1.0, 1.4) - 6.02, abs=0.2)
    assert shaping.duck(x, shaping.analyse(x), shaping.DUCK_FULL) is x                             # no voice: nothing to do
