"""The engine lives inside a worker process: it owns the models (loaded lazily, freed when the process exits)
and turns one stored clip into a stored preview."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import gpu
from .codec import SR, AacCodec, decode_file
from .config import DEFAULT_LOCAL_SPECIES, Config
from .locate import WIN
from .mixit import MIXIT_FILES, LazySeparator
from .perch import LABELS_FILE, OrtPerch
from .pipeline import Preview, PreviewError, finalize, make_preview
from .scoring import Hit, pick_alternatives, rank_species
from .species import Labels, LocalSpecies

log = logging.getLogger("kestrel_audio.engine")

CUDA_OVERHEAD_MIB = 160     # the CUDA context and cuDNN/cuBLAS handles sit outside onnxruntime's arena (measured 115-170 MiB)
RESIDENT_WEIGHTS_MIB = 520  # what stays on the GPU between runs: Perch 413 + both separators 76 + slack


class Engine:
    def __init__(self, cfg: Config, device: str):
        self.cfg, self.device = cfg, device
        self.labels = Labels.from_file(cfg.models_dir / LABELS_FILE)     # cheap: a text file
        self._perch: OrtPerch | None = None
        self._local: tuple[tuple[str, int, int], LocalSpecies] | None = None    # the species list in use, and the file stamp it was read at
        self._unusable: tuple[str, int, int] | None = None                      # stamp of a list that could not be read (warn once)
        self._separators: list[LazySeparator] | None = None
        self.codec = AacCodec()
        self.footprint = gpu.Footprint() if device == "cuda" else None    # remembers who used the GPU before we did; claim() pins our PID
        self.last_peak_mib: int | None = None

    # GPU budget (cfg.gpu_cap_mib, default 1500 MiB for the whole process). Between runs only the weights stay resident
    # (onnxruntime hands the working memory back after every run), and one model runs at a time, so the peak is
    # about: CUDA overhead + resident weights + the biggest transient. Each separator's arena cap is what is left of the
    # budget after the overhead and the resident weights; the work areas that are actually needed are smaller (a 9 s
    # clip needs about 550 MiB with 8 tracks, 400 MiB with 4).
    @property
    def separator_arena_mib(self) -> int:
        return max(256, self.cfg.gpu_cap_mib - CUDA_OVERHEAD_MIB - RESIDENT_WEIGHTS_MIB)

    @property
    def perch(self) -> OrtPerch:
        if self._perch is None:
            gpu_mode = self.device == "cuda"
            self._perch = OrtPerch(self.cfg.models_dir, device=self.device, threads=self.cfg.cpu_threads, vram_mib=self.cfg.perch_arena_mib,
                                   batch=self.cfg.perch_batch_gpu if gpu_mode else self.cfg.perch_batch_cpu)
            if self.footprint is not None:
                self.footprint.claim()          # our CUDA context exists now: learn which GPU process is us
        return self._perch

    @property
    def separators(self) -> list[LazySeparator]:
        if self._separators is None:
            self._separators = [
                LazySeparator(self.cfg.models_dir / fname, k, device=self.device, threads=self.cfg.cpu_threads,
                              vram_mib=self.separator_arena_mib)
                for k, fname in MIXIT_FILES.items() if (self.cfg.models_dir / fname).exists()]
        return self._separators

    def vram_mib(self) -> int | None:
        """GPU memory this process holds right now (None on the CPU path or when the driver cannot say)."""
        return self.footprint.mib() if self.footprint is not None else None

    def local_species(self) -> LocalSpecies | None:
        """The species that can be real here: the list Home Assistant sent, else the built-in one. The file is looked at again before every
        job, so a new list counts from the next clip without restarting the worker. None when no list can be read (no alternatives)."""
        for path in (self.cfg.local_species_path, DEFAULT_LOCAL_SPECIES):
            try:
                st = path.stat()
            except OSError:
                continue
            stamp = (str(path), st.st_mtime_ns, st.st_size)
            if self._local is not None and self._local[0] == stamp:
                return self._local[1]
            try:
                local = LocalSpecies.from_file(self.labels, path)
            except (OSError, ValueError) as exc:                  # a broken file must not stop previews; the next list in line is used
                if self._unusable != stamp:
                    log.warning("species list %s is not usable (%s)", path, exc)
                    self._unusable = stamp
                continue
            self._local = (stamp, local)
            return local
        return None

    def alternatives(self, prev: Preview, local: LocalSpecies, idx: int | None, species: str | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None] | None:
        """What Perch makes of the whole clip besides the one species BirdNET-Go named: up to `cfg.alternatives` others it hears more strongly
        than that one, and its own score for the named species. None when there is nothing to say (no survey, or no local species)."""
        if prev.survey is None:
            return None
        local = local.including(idx)
        hits = rank_species(prev.survey, local)
        if not hits:
            return None
        alts, named = pick_alternatives(hits, idx, self.cfg.alternatives)

        def describe(h: Hit) -> dict[str, Any]:
            scientific = self.labels.names[h.index]
            return {"species": local.common.get(h.index) or scientific, "scientific": scientific, "score": round(h.score, 4), "raw": round(h.raw, 4),
                    "windowsHigh": h.windows_high, "window": {"start": round(h.start, 2), "end": round(h.start + WIN, 2)}}

        announced = None if named is None else {**describe(named), "species": species or describe(named)["species"], "rank": named.rank}
        return [describe(h) for h in alts], announced

    def process(self, *, clip_path: str, out_path: str, scientific: str | None, species: str | None) -> dict[str, Any]:
        t0 = time.perf_counter()
        x22 = decode_file(clip_path, SR, max_seconds=self.cfg.max_clip_s + 1.0)
        if len(x22) / SR > self.cfg.max_clip_s:
            raise PreviewError(f"the clip is longer than {self.cfg.max_clip_s:.0f} s")
        idx = self.labels.find(scientific)
        cleanup = self.device == "cuda" or self.cfg.cpu_cleanup
        local = self.local_species() if self.cfg.alternatives > 0 else None
        t_dec = time.perf_counter() - t0
        scorer = self.perch if idx is not None else None
        # One Perch pass over the whole clip serves the search for the matched moment AND the alternatives. Without a species list, and
        # for a clip too short to search, there is nothing to ask for.
        surveyor = self.perch.survey if (local is not None or (idx is not None and len(x22) / SR > WIN)) else None
        with gpu.PeakSampler(self.footprint) as peak:
            prev = make_preview(x22, species_idx=idx, scorer=scorer,
                                separators=self.separators if (cleanup and idx is not None) else [], cleanup=cleanup, surveyor=surveyor)
            shipped, prev = finalize(prev, codec=self.codec, scorer=scorer, species_idx=idx)
        self.last_peak_mib = peak.peak
        out = Path(out_path)
        tmp = out.with_suffix(".part")
        tmp.write_bytes(shipped.data)
        tmp.replace(out)
        seg = prev.segment
        tm = {k: round(v, 3) for k, v in prev.timings.items()}
        tm["decodeS"] = round(t_dec, 3)
        tm["totalS"] = round(time.perf_counter() - t0, 3)
        notes = list(prev.notes)
        if idx is None:
            notes.append("no scientific name Perch knows" if scientific else "no scientific name was given")
        said = self.alternatives(prev, local, idx, species) if local is not None else None
        if self.cfg.alternatives > 0 and local is None:
            notes.append("no species list could be read, so no alternatives")
        alternatives, announced = said if said is not None else (None, None)
        return {
            "segment": {"start": round(seg.start, 2), "end": round(seg.end, 2), "source": seg.source},
            "method": prev.method,
            "variant": prev.variant,
            "cleaned": prev.cleaned,
            "scores": None if prev.score_original is None else {"original": round(prev.score_original, 4), "preview": round(prev.score_preview, 4)},
            "scoresNative": None if prev.native_original is None else {"original": round(prev.native_original, 4), "preview": round(prev.native_preview, 4)},
            "bestWindow": {"start": seg.best_start, "confidence": round(seg.best_conf, 4)} if seg.source == "perch" else None,
            "alternatives": alternatives,
            "announced": announced,
            "localSpecies": None if local is None else local.summary(),
            "loudnessLufs": round(shipped.lufs, 2),
            "truePeakDbtp": round(shipped.true_peak_db, 2),
            "durationS": round(shipped.duration_s, 2),
            "notes": notes,
            "decision": prev.decision,
            "timings": tm,
            "bytes": len(shipped.data),
            "completedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
