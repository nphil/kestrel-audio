"""The verification gate: when is a clean-up allowed to replace the untouched moment? Pure logic, no models.

A clean-up candidate is offered to the listener only if ALL of these hold:
  * it makes the animal stand out at least 3 dB more (clarity) and cuts the background at least 3 dB,
  * its loudness normalisation was clean (no limiter crushing, on target),
  * Perch is at least as sure of the species as on the untouched moment, minus 0.02, measured twice:
    at the original volume ("native") and at the loud preview volume.
Among candidates that pass, the one with the highest clarity wins. If none pass, the untouched moment is sent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

TOLERANCE = 0.02            # Perch confidence may be this much below the untouched moment's
MIN_CLARITY_GAIN_DB = 3.0
MIN_SUPPRESSION_DB = 3.0
CHUNK = 4                   # candidates scored per round (a round is one batched Perch call)

SOURCE_TAU = 0.30           # an AI source "matches" if its species score >= 30% of the best source's ...
SOURCE_TAU_MIN = 0.02       # ... and at least this much in absolute terms


@dataclass(frozen=True)
class Reference:
    """The untouched matched moment (candidate B)."""
    native_score: float     # Perch(species) on the raw segment at its original volume
    loud_score: float       # Perch(species) on the segment made loud (-16 LUFS)
    contrast_db: float      # clarity of the loud segment


@dataclass
class Candidate:
    id: str
    contrast_db: float
    suppression_db: float   # how much quieter the background got, measured at the original volume
    norm_clean: bool        # loudness normalisation landed on target without crushing the audio
    native_score: float | None = None
    loud_score: float | None = None
    reason: str = ""        # why it was not picked (filled in by `choose`)

    def gain_db(self, ref: Reference) -> float:
        return self.contrast_db - ref.contrast_db


@dataclass(frozen=True)
class Choice:
    candidate: Candidate
    gain_db: float


@dataclass
class Decision:
    choice: Choice | None
    considered: list[Candidate] = field(default_factory=list)


def eligible(c: Candidate, ref: Reference, *, min_gain: float = MIN_CLARITY_GAIN_DB, min_supp: float = MIN_SUPPRESSION_DB) -> str | None:
    """None when the candidate may be scored, else the reason it is skipped."""
    if not c.norm_clean:
        return "loudness could not be normalised cleanly"
    if c.gain_db(ref) < min_gain:
        return f"clarity gain {c.gain_db(ref):.1f} dB < {min_gain:g} dB"
    if c.suppression_db < min_supp:
        return f"background cut {c.suppression_db:.1f} dB < {min_supp:g} dB"
    return None


def native_ok(score: float, ref: Reference, tol: float = TOLERANCE) -> bool:
    return score >= ref.native_score - tol


def loud_ok(score: float, ref: Reference, tol: float = TOLERANCE) -> bool:
    return score >= ref.loud_score - tol


def choose(cands: Sequence[Candidate], ref: Reference,
           score_native: Callable[[list[Candidate]], dict[str, float]],
           score_loud: Callable[[list[Candidate]], dict[str, float]],
           *, chunk: int = CHUNK, tol: float = TOLERANCE) -> Decision:
    """Pick the cleanest verified candidate, spending as few Perch calls as possible.

    Candidates are tried from clearest to least clear, `chunk` at a time (each round = one batched Perch call for
    the native check, then one for the loud check of the survivors). The first round that holds a verified
    candidate decides, because every candidate in earlier rounds was clearer and failed: the result is exactly
    "the clearest verified candidate overall", at a fraction of the cost.

    `score_native` / `score_loud` receive candidates and return {candidate.id: Perch(species) confidence}.
    """
    dec = Decision(choice=None)
    pool: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.contrast_db, reverse=True):
        why = eligible(c, ref)
        dec.considered.append(c)
        if why:
            c.reason = why
        else:
            pool.append(c)
    for i in range(0, len(pool), chunk):
        batch = pool[i:i + chunk]
        nat = score_native(batch)
        survivors = []
        for c in batch:
            c.native_score = nat[c.id]
            if native_ok(c.native_score, ref, tol):
                survivors.append(c)
            else:
                c.reason = f"Perch {c.native_score:.3f} < {ref.native_score - tol:.3f} at the original volume"
        if not survivors:
            continue
        loud = score_loud(survivors)
        winners = []
        for c in survivors:
            c.loud_score = loud[c.id]
            if loud_ok(c.loud_score, ref, tol):
                winners.append(c)
            else:
                c.reason = f"Perch {c.loud_score:.3f} < {ref.loud_score - tol:.3f} at the preview volume"
        if winners:
            best = max(winners, key=lambda c: c.contrast_db)
            for c in winners:
                if c is not best:
                    c.reason = "verified, but a clearer candidate was too"
            dec.choice = Choice(best, best.gain_db(ref))
            return dec
    return dec


@dataclass(frozen=True)
class SourceChoice:
    top: int                  # index of the source Perch is surest of
    passing: tuple[int, ...]  # every source that also matches (always includes `top`)
    note: str                 # "" or why the pick is weak


def select_sources(species_conf: Sequence[float], any_conf: Sequence[float], *, tau: float = SOURCE_TAU,
                   tau_min: float = SOURCE_TAU_MIN) -> SourceChoice:
    """Which separated tracks hold the animal? `species_conf[k]` is the best Perch score of the detected species in
    track k, `any_conf[k]` the best score of any species (used only when no track scored the species at all)."""
    k = len(species_conf)
    if max(species_conf) <= 0.0:
        top = max(range(k), key=lambda i: any_conf[i])
        return SourceChoice(top, (top,), "no track scored the detected species; kept the one Perch is surest of overall")
    top = max(range(k), key=lambda i: species_conf[i])
    thr = max(tau * species_conf[top], tau_min)
    passing = tuple(i for i in range(k) if species_conf[i] >= thr) or (top,)
    return SourceChoice(top, passing, "")
