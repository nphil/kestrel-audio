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


def test_gain_cap_is_reported_when_the_limiter_would_have_to_crush_the_audio():
    sparse = np.zeros(6 * SR, dtype=np.float32)
    sparse[:SR // 4] = 0.9 * np.sin(2 * np.pi * 800 * np.arange(SR // 4) / SR)     # one loud burst, near-silence elsewhere
    rng = np.random.default_rng(1)
    sparse[SR:] = 1e-3 * rng.standard_normal(len(sparse) - SR)
    _, info = L.normalize(sparse, SR, target=-5.0)
    assert not L.loudness_clean(info)


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
