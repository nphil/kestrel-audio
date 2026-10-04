"""Perch v2 class labels: scientific name -> model class index, and the short list of classes that can be real here."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NUM_CLASSES = 14795   # Perch v2 output width (the label file adds one header line)
MAX_LOCAL_ENTRIES = 5000
MAX_NAME_LEN = 120
MAX_SOURCE_LEN = 300


def normalise(name: str) -> str:
    """'Cyanocitta_cristata ', 'cyanocitta  cristata' -> 'cyanocitta cristata'."""
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip().lower()


def is_sound_event(label: str) -> bool:
    """Perch v2 was trained on iNaturalist species plus 200 FSD50K sound events ('Wind', 'Rain', 'Chirp_and_tweet', ...). A
    species label is 'Genus species'; an event has no space or has underscores."""
    return " " not in label.strip() or "_" in label


class Labels:
    def __init__(self, names: list[str]):
        self.names = names
        self._index: dict[str, int] = {}
        for i, n in enumerate(names):
            self._index.setdefault(normalise(n), i)
        self._events: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.names)

    @classmethod
    def from_file(cls, path: str | Path, expected: int = NUM_CLASSES) -> "Labels":
        lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) == expected + 1:      # first line names the training set, not a class
            lines = lines[1:]
        if len(lines) != expected:
            raise ValueError(f"{path}: {len(lines)} labels, the model has {expected} classes")
        return cls(lines)

    def find(self, scientific: str | None) -> int | None:
        """Class index for a scientific name, or None if Perch does not know it. Subspecies ('Genus species sub') match
        on their first two words."""
        if not scientific or not scientific.strip():
            return None
        key = normalise(scientific)
        if key in self._index:
            return self._index[key]
        parts = key.split(" ")
        if len(parts) > 2:
            return self._index.get(" ".join(parts[:2]))
        return None

    @property
    def events(self) -> np.ndarray:
        """Class indices of the sound-event classes (wind, rain, traffic, speech, ...), ascending."""
        if self._events is None:
            self._events = np.array([i for i, n in enumerate(self.names) if is_sound_event(n)], dtype=np.int64)
        return self._events


# ------------------------------------------------------------------------------------------------ the local list

@dataclass(frozen=True)
class LocalList:
    """A list of species that can be heard where the microphone is, as it arrives (not yet matched to Perch)."""
    entries: tuple[tuple[str, str | None], ...]     # (scientific name, common name or None)
    source: str
    updated_at: str | None


def _text(value: Any, what: str, limit: int = MAX_NAME_LEN) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{what} must be text")
    value = value.strip()
    if len(value) > limit:
        raise ValueError(f"{what} is longer than {limit} characters")
    return value or None


def parse_local_list(payload: Any) -> LocalList:
    """`{"source": str, "updatedAt": str, "species": [{"scientific": str, "common": str}, ...]}`, or just the list. An entry may also
    use BirdNET-Go's own keys (`scientificName`, `commonName`), so its range-filter answer can be handed over as it is."""
    source, updated = "", None
    if isinstance(payload, dict):
        source = _text(payload.get("source"), "source", MAX_SOURCE_LEN) or ""
        updated = _text(payload.get("updatedAt"), "updatedAt", MAX_SOURCE_LEN)
        payload = payload.get("species")
    if not isinstance(payload, list) or not payload:
        raise ValueError("species must be a non-empty list")
    if len(payload) > MAX_LOCAL_ENTRIES:
        raise ValueError(f"more than {MAX_LOCAL_ENTRIES} species")
    entries: dict[str, tuple[str, str | None]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("every species must be an object with a scientific name")
        scientific = _text(item.get("scientific", item.get("scientificName")), "scientific")
        if not scientific:
            raise ValueError("every species needs a scientific name")
        common = _text(item.get("common", item.get("commonName")), "common")
        entries.setdefault(normalise(scientific), (scientific, common))
    return LocalList(tuple(entries.values()), source, updated)


def local_list_json(local: LocalList) -> dict[str, Any]:
    return {"source": local.source, "updatedAt": local.updated_at,
            "species": [{"scientific": s, "common": c} for s, c in local.entries]}


@dataclass(frozen=True)
class LocalSpecies:
    """What Perch's scores are measured against: the classes that can be real at this place (the local species, and every
    sound-event class so that wind and traffic can soak up probability instead of the nearest bird), as Perch class indices."""
    members: np.ndarray             # local species + sound events + extras, ascending
    species: np.ndarray             # the species among them (what 'could also be' draws from), ascending
    common: dict[int, str]          # class index -> common name, where the list gave one
    total: int                      # names in the list
    matched: int                    # names Perch knows
    source: str
    updated_at: str | None

    @classmethod
    def build(cls, labels: Labels, local: LocalList) -> "LocalSpecies":
        found: dict[int, str | None] = {}
        for scientific, common in local.entries:
            idx = labels.find(scientific)
            if idx is not None and not is_sound_event(labels.names[idx]):
                found.setdefault(idx, common)
        species = np.array(sorted(found), dtype=np.int64)
        members = np.union1d(species, labels.events).astype(np.int64)
        return cls(members, species, {i: c for i, c in found.items() if c}, len(local.entries), len(found), local.source, local.updated_at)

    @classmethod
    def from_file(cls, labels: Labels, path: str | Path) -> "LocalSpecies":
        return cls.build(labels, parse_local_list(json.loads(Path(path).read_text(encoding="utf-8"))))

    def including(self, idx: int | None) -> "LocalSpecies":
        """The same list with one more species that has to be a contender: the one BirdNET-Go just heard, whether or not the local
        list knows it (a coyote, an out-of-range visitor). Without it a real but unlisted animal could never win."""
        if idx is None or np.any(self.species == idx):
            return self
        return LocalSpecies(np.union1d(self.members, [idx]).astype(np.int64), np.union1d(self.species, [idx]).astype(np.int64),
                            self.common, self.total, self.matched, self.source, self.updated_at)

    def summary(self) -> dict[str, Any]:
        return {"count": self.total, "matched": self.matched, "source": self.source, "updatedAt": self.updated_at}
