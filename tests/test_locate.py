"""Where the matched moment is: window grid, best run, 8 s cap, margins and the fallbacks."""
import pytest

from kestrel_audio.locate import fallback_segment, pick_segment, whole_clip_segment, window_starts


def curve(clip, peaks, default=0.0):
    """{start: conf} for the full windows of a clip; `peaks` overrides specific starts."""
    starts = window_starts(clip, full_only=True)
    return starts, [peaks.get(s, default) for s in starts]


def test_window_grid_for_a_15_second_clip():
    full = window_starts(15.0, full_only=True)
    assert full[0] == 0.0 and full[-1] == 10.0 and len(full) == 21
    partial = window_starts(15.0, full_only=False)           # also windows holding >= 4.5 s of audio
    assert partial[-1] == 10.5 and len(partial) == 22


def test_window_grid_for_short_arrays():
    assert window_starts(6.0, full_only=False) == [0.0, 0.5, 1.0, 1.5]
    assert window_starts(6.0, full_only=True) == [0.0, 0.5, 1.0]
    assert window_starts(5.5, full_only=False) == [0.0, 0.5, 1.0]
    assert window_starts(3.0, full_only=True) == []


def test_single_peak_gives_the_run_plus_margin():
    starts, conf = curve(15.0, {5.5: 0.87, 6.0: 0.95, 6.5: 0.5})
    seg = pick_segment(starts, conf, 15.0)
    assert seg.best_start == 6.0 and seg.best_conf == 0.95 and seg.source == "perch"
    # run = windows 5.5 and 6.0 (both >= 90% of the best): 5.5 .. 11.0, half a second of margin each side
    assert (seg.run_start, seg.run_end) == (5.5, 11.0)
    assert (seg.start, seg.end) == (5.0, 11.5)


def test_margin_never_leaves_the_clip():
    starts, conf = curve(15.0, {0.0: 0.9})
    assert pick_segment(starts, conf, 15.0).start == 0.0
    starts, conf = curve(15.0, {10.0: 0.9})
    assert pick_segment(starts, conf, 15.0).end == 15.0


def test_a_long_plateau_is_capped_at_eight_seconds_around_the_best_window():
    starts, conf = curve(15.0, {s: 0.9 for s in window_starts(15.0, full_only=True)})
    conf[window_starts(15.0, full_only=True).index(7.0)] = 0.95
    seg = pick_segment(starts, conf, 15.0)
    assert seg.run_end - seg.run_start == 15.0
    assert seg.length <= 8.0 + 2 * 0.5 + 1e-9
    assert seg.start < 7.0 + 2.5 < seg.end                    # still holds the middle of the best window
    assert seg.windows_high == 21


def test_the_middle_of_several_equally_good_windows_is_the_best_one():
    starts, conf = curve(15.0, {4.0: 0.8, 4.5: 0.8, 5.0: 0.8})
    assert pick_segment(starts, conf, 15.0).best_start == 4.5


def test_a_quiet_neighbour_does_not_extend_the_run():
    starts, conf = curve(15.0, {6.0: 0.9, 6.5: 0.5, 7.0: 0.9})
    seg = pick_segment(starts, conf, 15.0)
    assert (seg.run_start, seg.run_end) == (6.0, 11.0) or (seg.run_start, seg.run_end) == (7.0, 12.0)
    assert seg.windows_high == 1


def test_species_never_scored_means_no_segment():
    starts, conf = curve(15.0, {})
    assert pick_segment(starts, conf, 15.0) is None
    assert pick_segment([], [], 15.0) is None


def test_fallback_uses_birdnet_gos_own_window():
    seg = fallback_segment(15.0)
    assert (seg.start, seg.end, seg.source) == (2.5, 8.5, "fallback")
    short = fallback_segment(6.0)
    assert short.start >= 0.0 and short.end <= 6.0 and short.length > 4.5


def test_clip_no_longer_than_one_window_is_used_whole():
    seg = whole_clip_segment(4.2)
    assert (seg.start, seg.end, seg.source) == (0.0, 4.2, "whole")
