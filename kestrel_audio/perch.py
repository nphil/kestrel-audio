"""Perch v2 through onnxruntime: CUDA (arena capped, so the process stays small on the shared GPU) or CPU."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .scoring import WindowScores, make_windows, to_scores
from .species import Labels

MODEL_FILE = "perch_v2_no_dft_fp32.onnx"     # FP32 only: the P40's FP16 rate is 1/64
LABELS_FILE = "perch_v2_labels.txt"


def make_session(path: str | Path, *, device: str, threads: int, vram_mib: int):
    """An onnxruntime session on `device` ("cuda" caps the CUDA arena at `vram_mib`; "cpu" uses `threads`).

    CUDA settings follow the measurements on the P40: no memory-pattern planning (it runs out of arena as soon as a second
    input length arrives), exact-size arena allocations, heuristic cuDNN algorithm choice (the exhaustive search re-runs for
    every new length). Pair it with `run_options(device)` so the arena is handed back to the GPU after every run."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False
    if device == "cuda":
        so.intra_op_num_threads = 2
        providers: list = [("CUDAExecutionProvider", {
            "device_id": 0,
            "gpu_mem_limit": int(vram_mib) * 1024 * 1024,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
        })]
    else:
        so.intra_op_num_threads = max(1, threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), so, providers=providers)
    if device == "cuda" and "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError("CUDAExecutionProvider did not load (providers: %s)" % sess.get_providers())
    return sess


def run_options(device: str):
    """Per-run options: on the GPU, shrink the arena after the run so the working memory goes back to the card between
    calls. Only the model weights stay resident, which is what lets Perch and both separators share a small budget."""
    if device != "cuda":
        return None
    import onnxruntime as ort

    ro = ort.RunOptions()
    ro.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    return ro


class OrtPerch:
    def __init__(self, models_dir: str | Path, *, device: str, threads: int, vram_mib: int, batch: int = 8):
        d = Path(models_dir)
        self.labels = Labels.from_file(d / LABELS_FILE)
        self.sess = make_session(d / MODEL_FILE, device=device, threads=threads, vram_mib=vram_mib)
        self.device = device
        self.batch = max(1, batch)
        self._inp = self.sess.get_inputs()[0].name
        self._ro = run_options(device)

    def logits(self, windows: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(windows), self.batch):
            out.append(self.sess.run(["label"], {self._inp: np.ascontiguousarray(windows[i:i + self.batch])}, self._ro)[0])
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), dtype=np.float32)

    def score(self, arrays: Sequence[np.ndarray], species_idx: int, *, full_only: bool = False) -> list[WindowScores]:
        """Score several arrays in one batched pass."""
        parts = [make_windows(a, full_only=full_only) for a in arrays]
        big = np.concatenate([w for _, w in parts]) if parts else np.zeros((0, 1), dtype=np.float32)
        logits = self.logits(big)
        res, at = [], 0
        for starts, w in parts:
            res.append(to_scores(starts, logits[at:at + len(w)], species_idx))
            at += len(w)
        return res
