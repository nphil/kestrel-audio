"""Perch v2 class labels: scientific name -> model class index."""
from __future__ import annotations

import re
from pathlib import Path

NUM_CLASSES = 14795   # Perch v2 output width (the label file adds one header line)


def normalise(name: str) -> str:
    """'Cyanocitta_cristata ', 'cyanocitta  cristata' -> 'cyanocitta cristata'."""
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip().lower()


class Labels:
    def __init__(self, names: list[str]):
        self.names = names
        self._index: dict[str, int] = {}
        for i, n in enumerate(names):
            self._index.setdefault(normalise(n), i)

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
