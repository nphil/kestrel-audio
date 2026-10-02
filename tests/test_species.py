import pytest

from kestrel_audio.species import NUM_CLASSES, Labels, normalise


def labels(n=5, header=True):
    names = ["Cyanocitta cristata", "Strix varia", "Canis latrans", "Pseudacris crucifer", "Thryothorus ludovicianus"][:n]
    return names, (["inat2024_fsd50k"] if header else []) + names


def test_lookup_ignores_case_underscores_and_spacing():
    L = Labels(labels()[0])
    assert L.find("Cyanocitta cristata") == 0
    assert L.find("  strix_varia ") == 1
    assert L.find("CANIS  LATRANS") == 2


def test_subspecies_match_on_genus_and_species():
    assert Labels(labels()[0]).find("Strix varia georgica") == 1


def test_unknown_or_empty_names_give_none():
    L = Labels(labels()[0])
    assert L.find("Turdus nowhereus") is None and L.find("") is None and L.find(None) is None and L.find("Strix") is None


def test_label_file_header_line_is_dropped(tmp_path):
    names, lines = labels(3, header=True)
    p = tmp_path / "labels.txt"
    p.write_text("\n".join(lines) + "\n")
    L = Labels.from_file(p, expected=3)
    assert L.names == names and L.find("Canis latrans") == 2


def test_label_count_must_match_the_model(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("a\nb\nc\n")
    with pytest.raises(ValueError):
        Labels.from_file(p, expected=5)
    assert NUM_CLASSES == 14795 and normalise("A_b  C") == "a b c"
