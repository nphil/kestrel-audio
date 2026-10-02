"""The verification gate and the AI-track selection."""
import random

import pytest

from kestrel_audio.verify import (Candidate, Reference, choose, eligible, loud_ok, native_ok, select_sources)

REF = Reference(native_score=0.60, loud_score=0.50, contrast_db=20.0)


def cand(cid, gain=10.0, supp=10.0, clean=True):
    return Candidate(cid, REF.contrast_db + gain, supp, clean)


def scorer(table):
    """A scoring callback that records what it was asked about."""
    calls = []

    def fn(batch):
        calls.append([c.id for c in batch])
        return {c.id: table[c.id] for c in batch}
    fn.calls = calls
    return fn


def test_tolerance_is_two_hundredths_below_the_untouched_moment():
    assert native_ok(0.585, REF) and not native_ok(0.575, REF)
    assert loud_ok(0.485, REF) and not loud_ok(0.475, REF)


def test_candidates_that_do_not_help_enough_are_not_even_scored():
    assert eligible(cand("a", gain=2.9), REF) is not None
    assert eligible(cand("a", gain=3.0), REF) is None
    assert eligible(cand("a", supp=2.0), REF) is not None
    assert eligible(cand("a", clean=False), REF) is not None
    nat, loud = scorer({}), scorer({})
    dec = choose([cand("a", gain=1.0), cand("b", clean=False)], REF, nat, loud)
    assert dec.choice is None and nat.calls == [] and loud.calls == []
    assert all(c.reason for c in dec.considered)


def test_the_clearest_verified_candidate_wins_even_when_gentler_ones_also_pass():
    cs = [cand("gentle", gain=5), cand("strong", gain=12), cand("medium", gain=8)]
    nat = scorer({"gentle": 0.6, "strong": 0.6, "medium": 0.6})
    loud = scorer({"gentle": 0.5, "strong": 0.5, "medium": 0.5})
    assert choose(cs, REF, nat, loud).choice.candidate.id == "strong"


def test_needs_both_volumes_a_native_pass_alone_is_not_enough():
    cs = [cand("x", gain=9)]
    dec = choose(cs, REF, scorer({"x": 0.6}), scorer({"x": 0.3}))
    assert dec.choice is None and "preview volume" in cs[0].reason
    dec = choose([cand("y", gain=9)], REF, scorer({"y": 0.3}), scorer({"y": 0.5}))
    assert dec.choice is None


def test_a_clearer_candidate_that_fails_is_skipped_for_a_gentler_one_that_passes():
    cs = [cand("harsh", gain=20), cand("ok", gain=6)]
    nat = scorer({"harsh": 0.1, "ok": 0.6})
    loud = scorer({"ok": 0.5})
    dec = choose(cs, REF, nat, loud, chunk=1)
    assert dec.choice.candidate.id == "ok" and dec.choice.gain_db == pytest.approx(6.0)
    assert nat.calls == [["harsh"], ["ok"]] and loud.calls == [["ok"]]


def test_scoring_stops_at_the_first_round_that_holds_a_verified_candidate():
    cs = [cand(f"c{i}", gain=20 - i) for i in range(8)]
    table = {c.id: 0.6 for c in cs}
    nat, loud = scorer(table), scorer({c.id: 0.5 for c in cs})
    dec = choose(cs, REF, nat, loud, chunk=3)
    assert dec.choice.candidate.id == "c0"
    assert len(nat.calls) == 1 and len(loud.calls) == 1        # one batched call each, not eight


def test_nothing_verified_means_the_untouched_moment_is_sent():
    cs = [cand(f"c{i}", gain=10 + i) for i in range(5)]
    dec = choose(cs, REF, scorer({c.id: 0.0 for c in cs}), scorer({}))
    assert dec.choice is None and len(dec.considered) == 5


def test_early_stopping_gives_exactly_the_brute_force_answer():
    rnd = random.Random(7)
    for _ in range(300):
        cs = [cand(f"c{i}", gain=rnd.uniform(0, 25), supp=rnd.uniform(0, 20), clean=rnd.random() > 0.1) for i in range(rnd.randint(0, 18))]
        nat_t = {c.id: rnd.uniform(0.3, 0.8) for c in cs}
        loud_t = {c.id: rnd.uniform(0.3, 0.7) for c in cs}
        ok = [c for c in cs if eligible(c, REF) is None and native_ok(nat_t[c.id], REF) and loud_ok(loud_t[c.id], REF)]
        want = max(ok, key=lambda c: c.contrast_db).id if ok else None
        got = choose(cs, REF, lambda b: {c.id: nat_t[c.id] for c in b}, lambda b: {c.id: loud_t[c.id] for c in b}, chunk=rnd.randint(1, 5))
        assert (got.choice.candidate.id if got.choice else None) == want


def test_source_selection_keeps_every_track_that_matches():
    ch = select_sources([0.05, 0.80, 0.30, 0.0], [0.2, 0.8, 0.4, 0.1])
    assert ch.top == 1 and ch.passing == (1, 2) and ch.note == ""     # 0.30 >= 0.3 * 0.80, 0.05 is not


def test_source_selection_has_an_absolute_floor():
    ch = select_sources([0.01, 0.015, 0.0, 0.0], [0.1, 0.1, 0.1, 0.1])
    assert ch.top == 1 and ch.passing == (1,)                         # both under the 0.02 floor -> only the best


def test_when_no_track_scores_the_species_the_surest_overall_track_is_used():
    ch = select_sources([0.0, 0.0, 0.0], [0.1, 0.6, 0.3])
    assert ch.top == 1 and ch.passing == (1,) and "no track" in ch.note
