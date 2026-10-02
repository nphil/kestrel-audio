"""The clean-up candidates tried for a matched moment.

Every candidate is built in the "scaled domain" (the moment pre-scaled to a 0.9 peak, which is also what the AI
separator sees). Names:

  G04 .. G20      spectral gate, strongest cut of 4 / 7 / 10 / 15 / 20 dB (hiss reduction only)
  M<k>            AI separation into k tracks (k = 4 or 8), keep only the track Perch is surest holds the animal
  M<k>S           keep every track that also matches (in case the animal was split across tracks)
  M<k>G, M4SG     the above plus a 12 dB gate pass
  MR<k>_50/25/12  "relaxed": keep the animal's track(s) at full level and turn the OTHER tracks down 6 / 12 / 18 dB
                  instead of deleting them (gentlest AI option; keeps the calls' quiet parts)
"""
from __future__ import annotations

import numpy as np

from . import dsp
from .verify import SourceChoice

GATES = {"G04": -4.0, "G07": -7.0, "G10": -10.0, "G15": -15.0, "G20": -20.0}
RELAX = {"50": 0.5, "25": 0.25, "12": 0.125}     # amplitude factors: -6, -12, -18 dB
POST_GATE_DB = -12.0

METHOD_TRIM, METHOD_GATE, METHOD_SEPARATE, METHOD_BOTH = "trim", "gate", "separate", "separate+gate"


def method_of(candidate_id: str) -> str:
    """The `method` reported in the API for a candidate id ("B" is the untouched moment)."""
    if candidate_id == "B":
        return METHOD_TRIM
    if candidate_id in GATES:
        return METHOD_GATE
    if candidate_id.endswith("G") or candidate_id == "M4SG":
        return METHOD_BOTH
    return METHOD_SEPARATE


def gate_variants(xs: np.ndarray, noise_lam: np.ndarray) -> dict[str, np.ndarray]:
    return {cid: dsp.wiener_dd(xs, floor_db=floor, lam=noise_lam) for cid, floor in GATES.items()}


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
