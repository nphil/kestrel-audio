"""The engine lives inside a worker process: it owns the models (loaded lazily, freed when the process exits)
and turns one stored clip into a stored preview."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import gpu
from .codec import SR, AacCodec, decode_file
from .config import Config
from .mixit import MIXIT_FILES, OrtMixit
from .perch import LABELS_FILE, OrtPerch
from .pipeline import PreviewError, finalize, make_preview
from .species import Labels


class Engine:
    def __init__(self, cfg: Config, device: str):
        self.cfg, self.device = cfg, device
        self.labels = Labels.from_file(cfg.models_dir / LABELS_FILE)     # cheap: a text file
        self._perch: OrtPerch | None = None
        self._separators: list[OrtMixit] | None = None
        self.codec = AacCodec()
        self.baseline_used_mib = (gpu.query().total_mib or 0) - (gpu.query().free_mib or 0) if device == "cuda" else 0

    # GPU memory plan (MiB), all inside the worker's cap: Perch holds its weights plus activations, each MixIT model is
    # loaded only if present. The numbers are tuned to the measured footprints (see README).
    @property
    def _caps(self) -> dict[str, int]:
        total = self.cfg.gpu_cap_mib
        return {"perch": int(total * 0.52), "mixit4": int(total * 0.18), "mixit8": int(total * 0.30)}

    @property
    def perch(self) -> OrtPerch:
        if self._perch is None:
            self._perch = OrtPerch(self.cfg.models_dir, device=self.device, threads=self.cfg.cpu_threads, vram_mib=self._caps["perch"])
        return self._perch

    @property
    def separators(self) -> list[OrtMixit]:
        if self._separators is None:
            seps = []
            for k, fname in MIXIT_FILES.items():
                p = self.cfg.models_dir / fname
                if p.exists():
                    seps.append(OrtMixit(p, k, device=self.device, threads=self.cfg.cpu_threads, vram_mib=self._caps[f"mixit{k}"]))
            self._separators = seps
        return self._separators

    def vram_mib(self) -> int | None:
        """GPU memory this process holds (None on CPU)."""
        if self.device != "cuda":
            return None
        own = gpu.used_by_pid_mib(os.getpid())
        if own is not None:
            return own
        info = gpu.query()
        if info.total_mib is None or info.free_mib is None:
            return None
        return max(0, (info.total_mib - info.free_mib) - self.baseline_used_mib)

    def process(self, *, clip_path: str, out_path: str, scientific: str | None, species: str | None) -> dict[str, Any]:
        t0 = time.perf_counter()
        x22 = decode_file(clip_path, SR, max_seconds=self.cfg.max_clip_s + 1.0)
        if len(x22) / SR > self.cfg.max_clip_s:
            raise PreviewError(f"the clip is longer than {self.cfg.max_clip_s:.0f} s")
        idx = self.labels.find(scientific)
        cleanup = self.device == "cuda" or self.cfg.cpu_cleanup
        t_dec = time.perf_counter() - t0
        scorer = self.perch if idx is not None else None
        prev = make_preview(x22, species_idx=idx, scorer=scorer,
                            separators=self.separators if (cleanup and idx is not None) else [], cleanup=cleanup)
        shipped, prev = finalize(prev, codec=self.codec, scorer=scorer, species_idx=idx)
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
        return {
            "segment": {"start": round(seg.start, 2), "end": round(seg.end, 2), "source": seg.source},
            "method": prev.method,
            "variant": prev.variant,
            "cleaned": prev.cleaned,
            "scores": None if prev.score_original is None else {"original": round(prev.score_original, 4), "preview": round(prev.score_preview, 4)},
            "scoresNative": None if prev.native_original is None else {"original": round(prev.native_original, 4), "preview": round(prev.native_preview, 4)},
            "bestWindow": {"start": seg.best_start, "confidence": round(seg.best_conf, 4)} if seg.source == "perch" else None,
            "loudnessLufs": round(shipped.lufs, 2),
            "truePeakDbtp": round(shipped.true_peak_db, 2),
            "durationS": round(shipped.duration_s, 2),
            "notes": notes,
            "decision": prev.decision,
            "timings": tm,
            "bytes": len(shipped.data),
            "completedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
