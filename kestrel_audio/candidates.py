"""The clean-up candidates tried for a matched moment.

Every candidate is built in the "scaled domain" (the moment pre-scaled to a 0.9 peak, which is also what the AI
separator sees). Names:

  G04 .. G20      spectral gate, strongest cut of 4 / 7 / 10 / 15 / 20 dB (hiss reduction only)
  N14 N08 N00     everything outside the animal's pitch range turned down, and the stretches between calls hushed by
                  14 / 8 / 0 dB (see shaping.py: nothing but turning down, so no gate or separator artefacts)
  W14 W08 W00     the same with the band one rung wider at each end
  H14 H08         only the hush between calls
  M<k>            AI separation into k tracks (k = 4 or 8), keep only the track Perch is surest holds the animal
  M<k>S           keep every track that also matches (in case the animal was split across tracks)
  M<k>G, M4SG     the above plus a 12 dB gate pass
  MR<k>_50/25/12  "relaxed": keep the animal's track(s) at full level and turn the OTHER tracks down 6 / 12 / 18 dB
                  instead of deleting them (gentlest AI option; keeps the calls' quiet parts)
  V2, V1          nothing but a human voice turned down: the whole stretch it is heard in (V2), or, when it speaks over
                  the animal, only the pitches outside the animal's range (V1). Whichever of the two Perch accepts is
                  also applied to every candidate above, so a clean-up never brings the voice back.
"""
from __future__ import annotations

import numpy as np

from . import dsp, shaping
from .verify import SourceChoice

GATES = {"G04": -4.0, "G07": -7.0, "G10": -10.0, "G15": -15.0, "G20": -20.0}
SHAPES = {"N14": ("narrow", 14.0), "N08": ("narrow", 8.0), "N00": ("narrow", 0.0),      # id -> (which band, hush in dB)
          "W14": ("wide", 14.0), "W08": ("wide", 8.0), "W00": ("wide", 0.0),
          "H14": (None, 14.0), "H08": (None, 8.0)}
VOICE = {"V2": shaping.DUCK_FULL, "V1": shaping.DUCK_OUTSIDE}                         # in order of preference
RELAX = {"50": 0.5, "25": 0.25, "12": 0.125}     # amplitude factors: -6, -12, -18 dB
POST_GATE_DB = -12.0

METHOD_TRIM, METHOD_GATE, METHOD_SHAPE, METHOD_VOICE, METHOD_SEPARATE, METHOD_BOTH = "trim", "gate", "shape", "voice", "separate", "separate+gate"


def method_of(candidate_id: str) -> str:
    """The `method` reported in the API for a candidate id ("B" is the untouched moment)."""
    if candidate_id == "B":
        return METHOD_TRIM
    if candidate_id in GATES:
        return METHOD_GATE
    if candidate_id in SHAPES:
        return METHOD_SHAPE
    if candidate_id in VOICE:
        return METHOD_VOICE
    if candidate_id.endswith("G") or candidate_id == "M4SG":
        return METHOD_BOTH
    return METHOD_SEPARATE


def gate_variants(xs: np.ndarray, noise_lam: np.ndarray) -> dict[str, np.ndarray]:
    return {cid: dsp.wiener_dd(xs, floor_db=floor, lam=noise_lam) for cid, floor in GATES.items()}


def shape_variants(scene: shaping.Scene) -> dict[str, np.ndarray]:
    """The turn-down-only candidates; the ones that need a band Perch did not grant are left out."""
    out: dict[str, np.ndarray] = {}
    for cid, (which, hush) in SHAPES.items():
        band = None if which is None else getattr(scene, which)
        if which is None or band is not None:
            out[cid] = shaping.render(scene, band=band, hush_db=hush)
    return out


def separated_variants(k: int, tracks: np.ndarray, choice: SourceChoice) -> dict[str, np.ndarray]:
    """Candidates built from one separation run. `tracks` is (k, n) in the scaled domain."""
    top = tracks[choice.top]
    keep = tracks[list(choice.passing)].sum(axis=0)
    rest = [i for i in range(k) if i not in choice.passing]
    out: dict[str, np.ndarray] = {f"M{k}": top, f"M{k}S": keep, f"M{k}G": dsp.wiener_dd(top, floor_db=POST_GATE_DB)}
    if k == 4:
        out["M4SG"] = dsp.wiener_dd(keep, floor_db=POST_GATE_DB)
    rest_sum = tracks[rest].sum(axis=0) if rest else None
    for tag, beta in RELAX.items():
        out[f"MR{k}_{tag}"] = keep + beta * rest_sum if rest_sum is not None else keep
    return out
