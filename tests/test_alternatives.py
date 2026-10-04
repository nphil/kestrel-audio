"""'Could also be': the local species list, Perch's scores measured against it, and which species are worth offering."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from kestrel_audio import config
from kestrel_audio.engine import Engine
from kestrel_audio.scoring import Survey, local_probabilities, pick_alternatives, rank_species, softmax
from kestrel_audio.species import NUM_CLASSES, Labels, LocalList, LocalSpecies, is_sound_event, parse_local_list

NAMES = ["Cyanocitta cristata", "Strix varia", "Canis latrans", "Pseudacris crucifer", "Thryothorus ludovicianus", "Wind", "Chirp_and_tweet",
         "Bubo virginianus", "Poecile carolinensis", "Turdus migratorius", "Rain"]
JAY, OWL, COYOTE, PEEPER, WREN, WIND, CHIRP, HORNED, CHICKADEE, ROBIN, RAIN = range(len(NAMES))
LOCAL = [("Cyanocitta cristata", "Blue Jay"), ("Strix varia", "Barred Owl"), ("Canis latrans", "Coyote"), ("Pseudacris crucifer", "Spring Peeper"),
         ("Thryothorus ludovicianus", "Carolina Wren"), ("Bubo virginianus", "Great Horned Owl"), ("Poecile carolinensis", None)]


def labels():
    return Labels(list(NAMES))


def local(extra=()):
    return LocalSpecies.build(labels(), LocalList(tuple(LOCAL) + tuple(extra), "test", "2026-10-03"))


def survey(rows, hop=0.5):
    """rows: one {class index: logit} per window; every other class sits at logit 0."""
    logits = np.zeros((len(rows), len(NAMES)), dtype=np.float32)
    for w, row in enumerate(rows):
        for c, v in row.items():
            logits[w, c] = v
    return Survey(np.arange(len(rows)) * hop, logits)


# ------------------------------------------------------------------------------------------------ the list

def test_sound_events_are_the_classes_that_are_not_genus_and_species():
    assert is_sound_event("Wind") and is_sound_event("Chirp_and_tweet") and is_sound_event("Bathtub_(filling_or_washing)")
    assert not is_sound_event("Cyanocitta cristata") and not is_sound_event("Pelophylax 'esculentus'")
    assert list(labels().events) == [WIND, CHIRP, RAIN]


def test_the_list_keeps_species_perch_knows_and_adds_every_sound_event():
    s = local([("Turdus nowhereus", "Nowhere Thrush"), ("Wind", "Wind")])       # unknown to Perch; and an event is never a species
    assert (s.total, s.matched) == (len(LOCAL) + 2, len(LOCAL))
    assert list(s.species) == [JAY, OWL, COYOTE, PEEPER, WREN, HORNED, CHICKADEE]
    assert list(s.members) == sorted([JAY, OWL, COYOTE, PEEPER, WREN, HORNED, CHICKADEE, WIND, CHIRP, RAIN])
    assert s.common[JAY] == "Blue Jay" and CHICKADEE not in s.common            # a name is kept only when the list gave one


def test_a_species_birdnet_go_named_always_competes_even_if_the_list_lacks_it():
    s = local()
    assert ROBIN not in s.members and ROBIN in s.including(ROBIN).members and ROBIN in s.including(ROBIN).species
    assert s.including(JAY) is s and s.including(None) is s


def test_list_formats():
    plain = parse_local_list([{"scientific": "Strix varia", "common": "Barred Owl"}])
    bng = parse_local_list({"source": "BirdNET-Go", "updatedAt": "2026-10-03T21:30:39-04:00",
                            "species": [{"label": "Strix varia_Barred Owl", "scientificName": "Strix varia", "commonName": "Barred Owl"}]})
    assert plain.entries == bng.entries == (("Strix varia", "Barred Owl"),)
    assert (bng.source, bng.updated_at) == ("BirdNET-Go", "2026-10-03T21:30:39-04:00")
    assert parse_local_list([{"scientific": "A b"}, {"scientific": " a  B "}]).entries == (("A b", None),)    # the same species once


@pytest.mark.parametrize("bad", [None, [], {}, {"species": []}, "Strix varia", [1], [{"common": "Barred Owl"}], [{"scientific": 5}],
                                 [{"scientific": "x" * 121}], [{"scientific": "A b"}] * 1 + [{"scientific": f"G s{i}"} for i in range(5000)]])
def test_unusable_lists_are_refused_with_a_reason(bad):
    with pytest.raises(ValueError):
        parse_local_list(bad)


def test_the_built_in_list_is_a_usable_species_list_with_common_names():
    built_in = parse_local_list(json.loads(config.DEFAULT_LOCAL_SPECIES.read_text(encoding="utf-8")))
    assert len(built_in.entries) > 300 and all(common for _, common in built_in.entries)
    assert ("Cyanocitta cristata", "Blue Jay") in built_in.entries and built_in.source


# ------------------------------------------------------------------------------------------------ scores against the list

def test_scores_are_measured_against_local_classes_only():
    # a non-local species (the robin) owns the window; the best local bird's plain share is tiny, but among local classes it is the answer
    rows = [{ROBIN: 9.0, WREN: 3.0, JAY: 1.0}] * 4
    raw = softmax(survey(rows).logits)
    q = local_probabilities(survey(rows).logits, local().members)
    j = list(local().members).index(WREN)
    assert q.shape == (4, len(local().members)) and np.allclose(q.sum(axis=1), 1.0)
    assert raw[0, WREN] < 0.01 and q[0, j] > 0.6
    # with every class as a member it is the plain softmax
    assert np.allclose(local_probabilities(survey(rows).logits, np.arange(len(NAMES))), raw)


def test_noise_soaks_up_the_probability_instead_of_the_nearest_bird():
    noise = survey([{WIND: 9.0, JAY: 2.0, OWL: 1.5}] * 5)
    hits = rank_species(noise, local())
    assert hits[0].index == JAY and hits[0].score < 0.01          # without Wind in the contest the jay would be 'sure'
    assert softmax(noise.logits)[0, JAY] == pytest.approx(hits[0].raw, abs=1e-6)


def test_ranking_gives_best_window_persistence_and_where_it_is():
    rows = [{}] * 4 + [{OWL: 6.0}] * 5 + [{}] * 3                  # the owl is heard in windows 4-8 (0.5 s hop)
    hits = rank_species(survey(rows), local())
    top = hits[0]
    assert top.index == OWL and top.rank == 1
    assert top.start == pytest.approx(3.0)                         # the middle one of the tied best windows (6th: 4+2)
    assert top.windows_high == 5
    assert top.score > 0.95 and top.raw == pytest.approx(softmax(survey(rows).logits)[6, OWL], abs=1e-6)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1)) and all(h.index not in (WIND, CHIRP, RAIN) for h in hits)


def test_a_one_window_blip_has_persistence_one():
    hits = rank_species(survey([{}] * 3 + [{WREN: 7.0}] + [{}] * 3), local())
    assert hits[0].index == WREN and hits[0].windows_high == 1 and hits[0].start == pytest.approx(1.5)


def test_nothing_to_rank_without_windows_or_species():
    assert rank_species(Survey(np.zeros(0), np.zeros((0, len(NAMES)), dtype=np.float32)), local()) == []
    assert rank_species(survey([{}]), LocalSpecies.build(labels(), LocalList((("Turdus nowhereus", None),), "t", None))) == []


# ------------------------------------------------------------------------------------------------ which ones are offered

def hits_for(rows):
    return rank_species(survey(rows), local())


def test_only_species_perch_hears_more_strongly_than_the_named_one_are_offered():
    hits = hits_for([{OWL: 5.0, JAY: 4.5, WREN: 1.0}] * 4)
    alts, named = pick_alternatives(hits, WREN, 3)                  # BirdNET-Go said Carolina Wren; Perch prefers the owl and the jay
    assert [h.index for h in alts] == [OWL, JAY] and named.index == WREN and named.rank == 3
    alts, named = pick_alternatives(hits, OWL, 3)                   # Perch hears the named owl best: nothing to doubt
    assert alts == [] and named.rank == 1


def test_a_species_needs_a_real_score_and_the_list_is_capped():
    hits = hits_for([{OWL: 4.0, JAY: 3.5, COYOTE: 3.2, PEEPER: 2.5, WREN: -3.0}] * 3)
    assert len(pick_alternatives(hits, WREN, 3)[0]) == 3 and len(pick_alternatives(hits, WREN, 2)[0]) == 2
    assert pick_alternatives(hits, WREN, 0)[0] == []
    assert all(h.score >= 0.15 for h in pick_alternatives(hits, WREN, 10)[0])
    assert pick_alternatives(hits, WREN, 10, min_score=0.9)[0] == []


def test_a_named_species_perch_has_no_class_for_still_gets_alternatives():
    hits = hits_for([{OWL: 5.0}] * 3)
    alts, named = pick_alternatives(hits, None, 3)
    assert [h.index for h in alts] == [OWL] and named is None


# ------------------------------------------------------------------------------------------------ the engine's side

@pytest.fixture
def engine(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    names = [f"Genus{i} species" for i in range(NUM_CLASSES - 3)] + ["Wind", "Rain", "Chirp_and_tweet"]
    names[10], names[11], names[12] = "Strix varia", "Bubo virginianus", "Cyanocitta cristata"
    (models / "perch_v2_labels.txt").write_text("inat2024_fsd50k\n" + "\n".join(names) + "\n")
    monkeypatch.setenv("KESTREL_AUDIO_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("KESTREL_AUDIO_MODELS", str(models))
    return Engine(config.load(), "cpu")


def test_the_engine_uses_the_built_in_list_until_one_is_sent_and_follows_changes(engine):
    first = engine.local_species()
    assert first is not None and first.source.startswith("Atlanta") and engine.local_species() is first        # read once, then remembered
    sent = engine.cfg.local_species_path
    sent.parent.mkdir(parents=True)
    sent.write_text(json.dumps({"source": "sent", "species": [{"scientific": "Strix varia", "common": "Barred Owl"}]}))
    second = engine.local_species()
    assert second.source == "sent" and list(second.species) == [10]
    sent.write_text("{not json")
    assert engine.local_species().source.startswith("Atlanta")                                                # a broken file falls back
    sent.unlink()
    assert engine.local_species().source.startswith("Atlanta")


def clip_survey(rows):
    logits = np.zeros((len(rows), NUM_CLASSES), dtype=np.float32)
    for w, row in enumerate(rows):
        for c, v in row.items():
            logits[w, c] = v
    return SimpleNamespace(survey=Survey(np.arange(len(rows)) * 0.5, logits))


def test_alternatives_and_the_announced_species_as_the_service_reports_them(engine):
    local = LocalSpecies.build(engine.labels, parse_local_list([{"scientific": "Strix varia", "common": "Barred Owl"},
                                                                 {"scientific": "Bubo virginianus", "common": "Great Horned Owl"},
                                                                 {"scientific": "Cyanocitta cristata", "common": "Blue Jay"}]))
    owls = {10: 6.0, 11: 5.5, 12: 0.5}                              # Barred Owl and Great Horned Owl in the middle windows; the jay is faint throughout
    prev = clip_survey([{12: 0.5}, owls, owls, owls, {12: 0.5}])
    alts, announced = engine.alternatives(prev, local, 12, "Blue Jay")      # BirdNET-Go said Blue Jay
    assert [a["species"] for a in alts] == ["Barred Owl", "Great Horned Owl"]
    top = alts[0]
    assert top["scientific"] == "Strix varia" and top["window"] == {"start": 1.0, "end": 6.0} and top["windowsHigh"] == 3
    assert 0.15 <= top["score"] <= 1 and 0 < top["raw"] < top["score"]
    assert (announced["species"], announced["scientific"], announced["rank"]) == ("Blue Jay", "Cyanocitta cristata", 3)
    assert announced["score"] < top["score"]
    assert engine.alternatives(SimpleNamespace(survey=None), local, 12, "Blue Jay") is None


def test_the_species_named_by_birdnet_go_is_a_contender_even_when_the_list_has_never_heard_of_it(engine):
    local = LocalSpecies.build(engine.labels, parse_local_list([{"scientific": "Strix varia", "common": "Barred Owl"}]))
    prev = clip_survey([{11: 8.0, 10: 2.0}] * 4)                    # a Great Horned Owl nobody listed, and it is what BirdNET-Go heard
    alts, announced = engine.alternatives(prev, local, 11, "Great Horned Owl")
    assert alts == [] and announced["rank"] == 1 and announced["score"] > 0.9
