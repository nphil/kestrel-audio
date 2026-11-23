"""Loudness, limiter, fades and the post-encode trim, including the clips so quiet they meter as minus infinity."""
import numpy as np
import pytest

from kestrel_audio import loudness as L

SR = 22050


def bursts(seconds=6.0, peak=0.5, seed=0):
    """Background hiss plus three short 'calls': a stand-in for a camera clip."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = 0.02 * rng.standard_normal(n)
    t = np.arange(int(0.4 * SR)) / SR
    for start in (1.0, 2.7, 4.2):
        i = int(start * SR)
        x[i:i + len(t)] += 0.9 * np.sin(2 * np.pi * 3000 * t) * np.hanning(len(t))
    return (x / np.max(np.abs(x)) * peak).astype(np.float32)


def test_normalize_reaches_target_loudness_and_true_peak():
    y, info = L.normalize(bursts(peak=0.05), SR)
    assert info.clean
    assert info.lufs_out == pytest.approx(-16.0, abs=0.3)
    assert L.true_peak_db(y) <= -1.0 + 0.02


def test_very_quiet_clip_meters_as_minus_infinity_until_prescaled():
    raw = bursts(peak=3e-5)                       # about -90 dBFS: BS.1770 ignores everything below -70 LUFS
    assert L.lufs(raw, SR) == float("-inf")
    scaled, scale = L.prescale(raw)
    assert scale > 1e4 and np.max(np.abs(scaled)) == pytest.approx(0.9, rel=1e-4)
    y, info = L.normalize(scaled, SR)
    assert info.clean and info.lufs_out == pytest.approx(-16.0, abs=0.5)
    assert L.true_peak_db(y) <= -1.0 + 0.02


def test_signal_that_cannot_be_metered_is_peak_normalised_instead():
    x = (0.9 * np.sin(2 * np.pi * 1000 * np.arange(int(0.3 * SR)) / SR)).astype(np.float32)   # shorter than one BS.1770 block
    assert L.lufs(x, SR) == float("-inf")
    y, info = L.normalize(x, SR)
    assert "peak-normalised" in info.note
    assert L.true_peak_db(y) == pytest.approx(-1.0, abs=0.3)


def test_digital_silence_is_reported_not_amplified():
    y, info = L.normalize(np.zeros(3 * SR, dtype=np.float32), SR)
    assert info.note == "silent" and not np.any(y)
    assert L.prescale(np.zeros(100, dtype=np.float32))[1] == 0.0


def test_limiter_holds_the_ceiling_on_hot_audio():
    hot = bursts(peak=0.5) * 8.0
    y, gr = L.limit_true_peak(np.clip(hot, -4, 4), SR, -1.0)
    assert gr > 12.0
    assert L.true_peak_db(y) <= -1.0 + 0.02


def test_release_matches_the_plain_per_sample_recursion():
    """The limiter's release is vectorised; it must equal r[i] = g[i] if g[i] < r[i-1] else a*r[i-1] + (1-a)*g[i]."""
    rng = np.random.default_rng(3)
    n = 12000
    g = np.ones(n)
    for c in rng.integers(100, n - 100, 25):
        w = int(rng.integers(5, 300))
        g[c:c + w] = np.minimum(g[c:c + w], rng.uniform(0.02, 0.95))
    a = float(np.exp(-1.0 / (0.05 * SR)))
    ref, cur = np.empty(n), 1.0
    for i in range(n):
        cur = g[i] if g[i] < cur else a * cur + (1 - a) * g[i]
        ref[i] = cur
    assert np.max(np.abs(L._release(g, a, block=1024) - ref)) < 1e-9


def chips(seconds=6.0, every=0.7, ms=60.0, seed=1):
    """Quiet hiss with a short loud chip now and then: a sparse call, far quieter on the meter than to the ear."""
    rng = np.random.default_rng(seed)
    x = 0.002 * rng.standard_normal(int(seconds * SR))
    t = np.arange(int(ms * 1e-3 * SR)) / SR
    for start in np.arange(0.4, seconds - 0.3, every):
        i = int(start * SR)
        x[i:i + len(t)] += 0.9 * np.sin(2 * np.pi * 4000 * t) * np.hanning(len(t))
    return x.astype(np.float32)


def faint_call(hiss, seconds=6.0, seed=2):
    """Three calls in hiss of a chosen level (RMS)."""
    rng = np.random.default_rng(seed)
    x = hiss * rng.standard_normal(int(seconds * SR))
    t = np.arange(int(0.4 * SR)) / SR
    for start in (1.0, 2.7, 4.2):
        i = int(start * SR)
        x[i:i + len(t)] += 0.45 * np.sin(2 * np.pi * 3000 * t) * np.hanning(len(t))
    return x.astype(np.float32)


def rms_db(x):
    return 10.0 * np.log10(float(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-30)


def test_sparse_calls_are_not_squashed_to_reach_the_meter_target():
    x = chips(every=0.35, ms=10.0)
    # what gaining all the way to -16 LUFS does to these notes: the limiter holds them down and the hiss comes up against them
    forced, gr = L.limit_true_peak(x * 10 ** ((-16.0 - L.lufs(x, SR)) / 20.0), SR)
    assert gr > 4.0 and L.pump_db(x, forced, SR) > 2.0
    y, info = L.normalize(x, SR)
    assert info.clean and info.held_db > 3.0 and info.lufs_out == pytest.approx(-16.0 - info.held_db, abs=0.01)
    assert L.pump_db(x, y, SR) <= L.PUMP_TOLERANCE_DB + 0.05          # the notes keep their height above the hiss
    assert -2.5 < L.true_peak_db(y) <= -1.0 + 0.02                    # and they do reach the ceiling


def test_pump_is_zero_for_a_plain_gain_and_counts_calls_that_were_held_down():
    x = chips().astype(np.float64)
    assert L.pump_db(x, x * 3.0, SR) == pytest.approx(0.0, abs=1e-6)
    held = x.copy()
    held[np.abs(x) > 0.05] *= 10 ** (-6.0 / 20.0)                       # every note 6 dB down, the hiss where it was
    assert L.pump_db(x, held, SR) == pytest.approx(6.0, abs=1.0)


def with_pop(x, at=3.5, peak=0.95, ms=2.0):
    y = np.array(x, dtype=np.float32)
    n = int(ms * 1e-3 * SR)
    i = int(at * SR)
    y[i:i + n] += (peak * np.hanning(n)).astype(np.float32)
    return y, i, n


def test_tame_turns_down_the_pop_and_nothing_else():
    x, i, n = with_pop(bursts(peak=0.2))
    y, down = L.tame(x, SR)
    assert down > 6.0
    assert np.max(np.abs(y)) <= L.sustained_peak(x, SR) * 10 ** (L.POP_HEADROOM_DB / 20.0) * 1.001
    away = np.ones(len(x), dtype=bool)
    away[i - 200:i + n + 200] = False                                   # more than 9 ms either side of it
    assert np.array_equal(y[away], x.astype(np.float64)[away])
    calm, none = L.tame(bursts(peak=0.2), SR)
    assert none == 0.0 and np.array_equal(calm, bursts(peak=0.2).astype(np.float64))


def test_one_pop_neither_sets_the_volume_nor_drags_the_level_down_after_it():
    plain, plain_info = L.normalize(bursts(peak=0.05), SR)
    x, i, n = with_pop(bursts(peak=0.05))
    y, info = L.normalize(x, SR)
    assert info.tamed_db > 6.0
    assert info.lufs_out == pytest.approx(plain_info.lufs_out, abs=0.5)       # as loud as the same moment without the pop
    after = slice(i + n + int(0.02 * SR), i + n + int(0.10 * SR))             # 20 to 100 ms after it: where a limiter would still be letting go
    assert rms_db(y[after]) == pytest.approx(rms_db(plain[after]), abs=0.5)
    # left to the limiter at the same gain, the pop drags the hiss behind it down by more than 2 dB
    squashed, gr = L.limit_true_peak(x.astype(np.float64) * 10 ** (plain_info.gain_db / 20.0), SR)
    assert gr > 6.0 and rms_db(squashed[after]) < rms_db(plain[after]) - 2.0


def test_a_pop_that_may_not_be_touched_is_the_peak_everything_else_is_measured_against():
    x, i, n = with_pop(bursts(peak=0.05))
    _, tamed = L.normalize(x, SR)
    y, info = L.normalize(x, SR, tame_pops=False)
    assert tamed.tamed_db > 6.0 and tamed.lufs_out == pytest.approx(-16.0, abs=0.5)
    assert info.tamed_db == 0.0 and info.max_gr_db <= L.CALL_LIMIT_DB + 0.5      # nothing but gain and a little limiting touched it
    assert info.held_db > 6.0 and info.lufs_out < tamed.lufs_out - 6.0           # the price: a quieter preview
    assert L.true_peak_db(y) <= -1.0 + 0.02


def with_chip(x, at=3.5, peak=0.95, hz=7000.0, ms=8.0):
    y = np.array(x, dtype=np.float32)
    t = np.arange(int(ms * 1e-3 * SR)) / SR
    i = int(at * SR)
    y[i:i + len(t)] += (peak * np.sin(2 * np.pi * hz * t) * np.hanning(len(t))).astype(np.float32)
    return y, i, len(t)


def with_thump(x, at=3.5, peak=0.95, hz=40.0):
    y = np.array(x, dtype=np.float32)
    n = int(SR / hz / 2)                                                  # half a cycle: 12 ms
    i = int(at * SR)
    y[i:i + n] += (peak * np.sin(np.pi * np.arange(n) / n)).astype(np.float32)
    return y


def test_a_low_thump_is_told_from_a_sharp_chip():
    low = with_thump(bursts(peak=0.05))
    assert L.is_thump(low, L.pop_gain(low, SR), SR)
    high, _, _ = with_chip(bursts(peak=0.05))
    assert not L.is_thump(high, L.pop_gain(high, SR), SR)
    assert L.pop_gain(bursts(peak=0.05), SR) is None                      # nothing stands out: nothing to ask about


def test_without_its_pop_the_moment_is_otherwise_untouched():
    x, i, n = with_chip(bursts(peak=0.05))
    gone = L.without_pop(x, L.pop_gain(x, SR), SR)
    assert np.max(np.abs(gone[i:i + n])) < 0.1 * np.max(np.abs(x[i:i + n]))
    away = np.ones(len(x), dtype=bool)
    away[i - 400:i + n + 400] = False                                     # 18 ms either side of it
    assert np.array_equal(gone[away], x[away])


def test_hiss_is_not_gained_past_the_background_ceiling():
    x = faint_call(hiss=0.05)
    y, info = L.normalize(x, SR)
    assert 0.5 < info.floor_cut_db < L.FLOOR_MAX_CUT_DB - 0.5
    assert L.floor_level_db(y, SR) == pytest.approx(L.FLOOR_CEILING_DB, abs=0.3)      # the background sits at the ceiling, not above it
    assert info.lufs_out == pytest.approx(-16.0 - info.floor_cut_db, abs=0.5) and info.held_db == pytest.approx(-16.0 - info.lufs_out)


def test_the_loudness_given_up_for_a_quiet_background_is_bounded():
    y, info = L.normalize(faint_call(hiss=0.2), SR)
    assert info.floor_cut_db == pytest.approx(L.FLOOR_MAX_CUT_DB)
    assert info.lufs_out == pytest.approx(-16.0 - L.FLOOR_MAX_CUT_DB, abs=0.5)
    quiet, quiet_info = L.normalize(faint_call(hiss=0.002), SR)                        # nothing to hold back: straight to the target
    assert quiet_info.floor_cut_db == 0.0 and quiet_info.lufs_out == pytest.approx(-16.0, abs=0.3)


def test_fade_is_silent_at_both_ends_and_leaves_the_middle_alone():
    y = L.fade(np.ones(SR, dtype=np.float32), SR, 75)
    assert y[0] == 0.0 and y[-1] < 1e-3 and y[SR // 2] == 1.0 and len(y) == SR
    assert np.all(np.diff(y[:int(0.075 * SR)]) >= 0)
    tiny = L.fade(np.ones(10, dtype=np.float32), SR, 75)    # never longer than a quarter of the clip
    assert len(tiny) == 10 and tiny[5] == 1.0


def _at_ceiling():
    y, _ = L.limit_true_peak(np.clip(bursts(peak=0.5) * 8.0, -4, 4), SR, -1.0)    # peak sits right at -1 dBTP
    assert L.true_peak_db(y) == pytest.approx(-1.0, abs=0.05)
    return y


def test_trim_lowers_gain_until_the_decoded_file_holds_the_ceiling():
    y = _at_ceiling()
    overshoot = lambda a: (a * 1.10).astype(np.float32)       # an "encoder" that comes back 0.8 dB hot
    _, dec, trim, tp = L.trim_to_true_peak(y, overshoot, SR)
    assert tp <= -1.0 + 0.02
    assert -1.6 < trim < -0.7
    assert L.true_peak_db(dec) == pytest.approx(tp)


def test_trim_does_nothing_when_the_decoded_peak_is_already_fine():
    _, _, trim, tp = L.trim_to_true_peak(_at_ceiling(), lambda a: a * 0.99, SR)
    assert trim == 0.0 and tp <= -1.0
