#!/usr/bin/env python3
"""Convert Google's bird MixIT separation checkpoints (TensorFlow 1 graphs) to ONNX.

    python tools/convert_mixit.py --sources 4 --out mixit4.onnx --download
    python tools/convert_mixit.py --sources 8 --out mixit8.onnx --ckpt-dir /path/to/output_sources8

The service runs the result with onnxruntime (CPU or CUDA), so TensorFlow is only needed here, in a build stage.

Upstream: https://github.com/google-research/sound-separation/tree/master/models/bird_mixit (code and checkpoints are
Apache-2.0, see THIRD_PARTY_NOTICES). Checkpoints: gs://gresearch/sound_separation/bird_mixit_model_checkpoints/
output_sources4|output_sources8, fetched over https://storage.googleapis.com/gresearch/... and verified against the sha256
pins below.

The model maps a 22.05 kHz mono waveform ``audio`` [1, 1, samples] to ``sources`` [1, K, samples] (K = 4 or 8); the time axis
is dynamic (any length, tested 5 s to 15 s).

Three things in the TF graph need care, and each is handled here:

1. tf2onnx 1.16 turns the integer ops FloorMod/FloorDiv into Div-based formulas that truncate toward zero. The graph pads
   its input to a multiple of the 11-sample frame hop with ``FloorMod(-n, 11)``, so the stock conversion silently runs with
   a negative pad (crops 3 samples) and fails at run time with a broadcast error whenever the length is not a multiple of
   11. Fixed with ONNX ``Mod(fmod=0)``, which has TF's floor semantics for integers.
2. The 28 dilated depthwise convolutions are written as SpaceToBatchND -> DepthwiseConv2dNative(VALID) -> BatchToSpaceND with
   paddings computed from the dynamic length. They are fused into one DepthwiseConv2dNative with ``dilations=[1, d, 1, 1]``
   and zero padding ``d`` on both sides. That is exactly what the padded batch-to-space dance computes (the S2B pads are
   [d, d + extra] and the B2S crops drop ``extra``), and it removes about 2000 reshape/transpose/pad nodes.
3. The result is checked against the original TF graph on synthetic audio of two different lengths before the file is kept.

Validated stack (Docker python:3.11-slim, see PINNED_REQUIREMENTS): tensorflow-cpu 2.12.1, tf2onnx 1.16.1, onnx 1.16.2,
onnxruntime 1.26.0, numpy 1.24.3, protobuf 3.20.3.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.request

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SAMPLE_RATE = 22050
DEFAULT_OPSET = 18
GCS_URI = "gs://gresearch/sound_separation/bird_mixit_model_checkpoints"
HTTPS_BASE = "https://storage.googleapis.com/gresearch/sound_separation/bird_mixit_model_checkpoints"

PINNED_REQUIREMENTS = (
    "tensorflow-cpu==2.12.1",
    "tf2onnx==1.16.1",
    "onnx==1.16.2",
    "onnxruntime==1.26.0",
    "numpy==1.24.3",
    "protobuf==3.20.3",
)

# name -> (bytes, sha256). The sha256 values were computed from the files downloaded from the URLs above; the md5 of the same
# files matches the md5Hash that the bucket reports.
CHECKPOINTS = {
    4: {
        "subdir": "output_sources4",
        "prefix": "model.ckpt-3223090",
        "files": {
            "inference.meta": (3194589, "cdc4187475d9b04ba2f25cb60d07276d2209f45dc13dddfb686085922339df31"),
            "model.ckpt-3223090.index": (17670, "d0f2a6eecdf6b6ed11c05a6d1a185dd2bc0158e53e808b8d6fe2ebacf7f05b85"),
            "model.ckpt-3223090.data-00000-of-00001": (110260324, "5f95e80d7075dad964554e39aeef706ab5290ee5d5cd7783cea9c36e36c97e42"),
        },
    },
    8: {
        "subdir": "output_sources8",
        "prefix": "model.ckpt-2178900",
        "files": {
            "inference.meta": (3194589, "7b3051302335f7536f2f8b628c60870c2148ddcadbb44f504016b72ab8adc428"),
            "model.ckpt-2178900.index": (17676, "6aba0125433ec2a593ce1c58ff02af991053815950af4a6a84bad5534a48bc3c"),
            "model.ckpt-2178900.data-00000-of-00001": (113406052, "13b367efadc2472fa5357a9f518fdcfe6eb76328ee5990b6e0c56ddb18b8e8ce"),
        },
    },
}

TF_INPUT = "input_audio/receiver_audio:0"
TF_OUTPUT = "denoised_waveforms:0"
ONNX_INPUT = "audio"
ONNX_OUTPUT = "sources"

# Self-check lengths: not multiples of the 11-sample hop or of any dilation, the case that broke the stock conversion.
SELF_CHECK_SAMPLES = (5 * SAMPLE_RATE + 3, 7 * SAMPLE_RATE + 7)
SELF_CHECK_MAX_ABS = 1e-3
SELF_CHECK_MIN_SDR_DB = 50.0


def log(msg: str) -> None:
    print(f"[convert_mixit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------------------------------
# checkpoint files
# --------------------------------------------------------------------------------------------------------------------
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_checkpoint(sources: int, directory: str) -> None:
    for name, (size, digest) in CHECKPOINTS[sources]["files"].items():
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise SystemExit(f"missing checkpoint file: {path}")
        if os.path.getsize(path) != size or sha256_of(path) != digest:
            raise SystemExit(f"{path} does not match the pinned upstream file (size {size}, sha256 {digest})")


def download_checkpoint(sources: int, directory: str, attempts: int = 3) -> None:
    os.makedirs(directory, exist_ok=True)
    spec = CHECKPOINTS[sources]
    for name, (size, digest) in spec["files"].items():
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.path.getsize(path) == size and sha256_of(path) == digest:
            log(f"have {name}")
            continue
        url = f"{HTTPS_BASE}/{spec['subdir']}/{name}"
        for attempt in range(1, attempts + 1):
            tmp = path + ".part"
            try:
                log(f"download {url} ({size / 1e6:.1f} MB)")
                h = hashlib.sha256()
                with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
                    for block in iter(lambda: resp.read(1 << 20), b""):
                        h.update(block)
                        out.write(block)
                if h.hexdigest() != digest or os.path.getsize(tmp) != size:
                    raise OSError(f"sha256/size mismatch for {url}")
                os.replace(tmp, path)
                break
            except OSError as exc:
                if os.path.exists(tmp):
                    os.remove(tmp)
                if attempt == attempts:
                    raise SystemExit(f"download failed: {exc}")
                log(f"retry {attempt}/{attempts - 1} after: {exc}")
                time.sleep(2 * attempt)


# --------------------------------------------------------------------------------------------------------------------
# TF graph: restore, freeze, rewrite dilated convolutions
# --------------------------------------------------------------------------------------------------------------------
def freeze_checkpoint(directory: str, prefix: str):
    """Restore the TF1 inference graph and return a GraphDef with the variables baked in as constants."""
    import tensorflow as tf

    tf1 = tf.compat.v1
    tf1.disable_v2_behavior()
    graph = tf1.Graph()
    with graph.as_default():
        saver = tf1.train.import_meta_graph(os.path.join(directory, "inference.meta"))
    with tf1.Session(graph=graph, config=tf1.ConfigProto(device_count={"GPU": 0})) as sess:
        with graph.as_default():
            saver.restore(sess, os.path.join(directory, prefix))
        return tf1.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [TF_OUTPUT.split(":")[0]])


def _node_name(tensor_name: str) -> str:
    return tensor_name.split(":")[0].lstrip("^")


def fuse_dilated_depthwise(graph_def):
    """SpaceToBatchND -> DepthwiseConv2dNative(VALID) -> BatchToSpaceND  =>  one dilated DepthwiseConv2dNative.

    Returns (new GraphDef, number of fused convolutions). The input GraphDef is not modified.
    """
    import tensorflow as tf

    tf1 = tf.compat.v1
    gd = tf1.GraphDef()
    gd.CopyFrom(graph_def)
    nodes = {n.name: n for n in gd.node}
    consumers: dict[str, list[str]] = {}
    for n in gd.node:
        for i in n.input:
            consumers.setdefault(_node_name(i), []).append(n.name)

    def const_value(tensor_name):
        node = nodes[_node_name(tensor_name)]
        if node.op != "Const":
            raise RuntimeError(f"{node.name} is not constant; this checkpoint does not have the expected graph layout")
        return tf.make_ndarray(node.attr["value"].tensor).tolist()

    rewire: dict[str, str] = {}
    drop: set[str] = set()
    for b2s in [n for n in gd.node if n.op == "BatchToSpaceND"]:
        dw = nodes[_node_name(b2s.input[0])]
        s2b = nodes[_node_name(dw.input[0])]
        if dw.op != "DepthwiseConv2dNative" or s2b.op != "SpaceToBatchND" or consumers.get(dw.name) != [b2s.name]:
            raise RuntimeError(f"unexpected graph around {b2s.name}: {s2b.op} -> {dw.op} -> {b2s.op}")
        block = const_value(s2b.input[1])
        if block != const_value(b2s.input[1]) or len(block) != 2 or block[1] != 1:
            raise RuntimeError(f"unexpected block shape at {b2s.name}: {block}")
        if dw.attr["padding"].s != b"VALID" or dw.attr["data_format"].s != b"NHWC":
            raise RuntimeError(f"unexpected conv padding/layout at {dw.name}")
        d = block[0]
        dw.input[0] = s2b.input[0]
        del dw.attr["dilations"].list.i[:]
        dw.attr["dilations"].list.i.extend([1, d, 1, 1])
        dw.attr["padding"].s = b"EXPLICIT"
        del dw.attr["explicit_paddings"].list.i[:]
        dw.attr["explicit_paddings"].list.i.extend([0, 0, d, d, 0, 0, 0, 0])
        rewire[b2s.name] = dw.name
        drop.update((b2s.name, s2b.name))

    for n in gd.node:
        for idx, inp in enumerate(n.input):
            target = rewire.get(_node_name(inp))
            if target:
                n.input[idx] = ("^" if inp.startswith("^") else "") + target
    fused = tf1.GraphDef()
    fused.node.extend(n for n in gd.node if n.name not in drop)
    fused.versions.CopyFrom(gd.versions)
    fused.library.CopyFrom(gd.library)
    # drops the now-unused paddings/crops subgraphs
    fused = tf1.graph_util.extract_sub_graph(fused, [TF_OUTPUT.split(":")[0]])
    return fused, len(rewire)


# --------------------------------------------------------------------------------------------------------------------
# tf2onnx
# --------------------------------------------------------------------------------------------------------------------
def _float_dtypes():
    from onnx import TensorProto

    return (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE)


def _floor_mod(ctx, node, name, args):
    """TF FloorMod (result takes the divisor's sign). Integers use ONNX Mod(fmod=0); tf2onnx's Div-based version truncates."""
    shapes, dtypes = node.output_shapes, node.output_dtypes
    if ctx.get_dtype(node.input[0]) in _float_dtypes():
        div = ctx.make_node("Div", list(node.input))
        floor = ctx.make_node("Floor", div.output)
        mul = ctx.make_node("Mul", [floor.output[0], node.input[1]])
        ctx.remove_node(node.name)
        ctx.make_node("Sub", [node.input[0], mul.output[0]], name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)
    else:
        ctx.remove_node(node.name)
        ctx.make_node("Mod", list(node.input), attr={"fmod": 0}, name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)


def _floor_div(ctx, node, name, args):
    """TF FloorDiv. Integers: (x - Mod(x, y)) / y is an exact division, so ONNX's truncating Div gives the floor."""
    shapes, dtypes = node.output_shapes, node.output_dtypes
    if ctx.get_dtype(node.input[0]) in _float_dtypes():
        div = ctx.make_node("Div", list(node.input))
        ctx.remove_node(node.name)
        ctx.make_node("Floor", div.output, name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)
    else:
        mod = ctx.make_node("Mod", list(node.input), attr={"fmod": 0})
        diff = ctx.make_node("Sub", [node.input[0], mod.output[0]])
        ctx.remove_node(node.name)
        ctx.make_node("Div", [diff.output[0], node.input[1]], name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)


def to_onnx(graph_def, opset: int):
    import tf2onnx

    model, _ = tf2onnx.convert.from_graph_def(
        graph_def,
        input_names=[TF_INPUT],
        output_names=[TF_OUTPUT],
        opset=opset,
        shape_override={TF_INPUT: [1, 1, None]},
        custom_op_handlers={"FloorMod": (_floor_mod, []), "FloorDiv": (_floor_div, [])},
    )
    return model


def finalize_model(model, sources: int, opset: int):
    """Friendly tensor names, named dynamic axis, metadata."""
    import onnx
    import tensorflow as tf
    import tf2onnx

    renames = {TF_INPUT: ONNX_INPUT, TF_OUTPUT: ONNX_OUTPUT}
    g = model.graph
    for node in g.node:
        for field in (node.input, node.output):
            for i, t in enumerate(field):
                if t in renames:
                    field[i] = renames[t]
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.name in renames:
            vi.name = renames[vi.name]
    for vi, k in ((g.input[0], 1), (g.output[0], sources)):
        dims = vi.type.tensor_type.shape.dim
        dims[0].dim_value, dims[1].dim_value = 1, k
        dims[2].dim_param = "samples"
    spec = CHECKPOINTS[sources]
    ckpt_id = f"bird_mixit/{spec['subdir']}/{spec['prefix']}"
    onnx.helper.set_model_props(
        model,
        {
            "sample_rate": str(SAMPLE_RATE),
            "num_sources": str(sources),
            "checkpoint": ckpt_id,
            "source": f"{GCS_URI}/{spec['subdir']} (https://github.com/google-research/sound-separation/tree/master/models/bird_mixit)",
            "license": "Apache-2.0, Copyright Google LLC (weights and code)",
            "input": f"{ONNX_INPUT}: float32 [1, 1, samples], mono, {SAMPLE_RATE} Hz",
            "output": f"{ONNX_OUTPUT}: float32 [1, {sources}, samples], same length as the input",
            "converter": f"kestrel-audio tools/convert_mixit.py; tensorflow {tf.__version__}, tf2onnx {tf2onnx.__version__}, "
            f"onnx {onnx.__version__}, opset {opset}",
        },
    )
    model.doc_string = f"Google bird MixIT {sources}-source separation model ({ckpt_id}) converted to ONNX."
    return model


# --------------------------------------------------------------------------------------------------------------------
# self-check against the original TF graph
# --------------------------------------------------------------------------------------------------------------------
def synthetic_audio(samples: int, seed: int):
    """Bird-ish test signal: two gliding tones with envelopes plus noise, peak 0.9, float32."""
    import numpy as np

    rng = np.random.RandomState(seed)
    t = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    sig = 0.05 * rng.randn(samples)
    for f0, f1, period in ((3200.0, 5200.0, 0.9), (6100.0, 4300.0, 1.3)):
        phase = (t % period) / period
        freq = f0 + (f1 - f0) * phase
        sig += 0.4 * np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE) * (np.sin(np.pi * phase) ** 2)
    return (0.9 * sig / np.max(np.abs(sig))).astype(np.float32)


def self_check(onnx_path: str, reference_graph_def, sources: int) -> None:
    import numpy as np
    import onnxruntime as ort
    import tensorflow as tf

    tf1 = tf.compat.v1
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    if [i.name for i in sess.get_inputs()] != [ONNX_INPUT] or [o.name for o in sess.get_outputs()] != [ONNX_OUTPUT]:
        raise SystemExit("unexpected input/output names in the converted model")
    graph = tf1.Graph()
    with graph.as_default():
        tf1.import_graph_def(reference_graph_def, name="")
    tf_sess = tf1.Session(graph=graph, config=tf1.ConfigProto(device_count={"GPU": 0}))
    tf_in, tf_out = graph.get_tensor_by_name(TF_INPUT), graph.get_tensor_by_name(TF_OUTPUT)
    for seed, samples in enumerate(SELF_CHECK_SAMPLES):
        x = synthetic_audio(samples, seed)[None, None, :]
        t0 = time.time()
        got = sess.run([ONNX_OUTPUT], {ONNX_INPUT: x})[0]
        ort_s = time.time() - t0
        if got.shape != (1, sources, samples):
            raise SystemExit(f"self-check: output shape {got.shape}, expected {(1, sources, samples)}")
        if not np.isfinite(got).all():
            raise SystemExit("self-check: non-finite values in the ONNX output")
        ref = tf_sess.run(tf_out, {tf_in: x})
        diff = got.astype(np.float64) - ref.astype(np.float64)
        max_abs = float(np.abs(diff).max())
        sdr = [
            10 * np.log10(float(np.sum(ref[0, k].astype(np.float64) ** 2)) / max(float(np.sum(diff[0, k] ** 2)), 1e-30))
            for k in range(sources)
            if float(np.sum(ref[0, k].astype(np.float64) ** 2)) > 1e-8
        ]
        log(f"self-check T={samples} ({samples / SAMPLE_RATE:.2f}s): shape ok, finite, max|onnx-tf|={max_abs:.2e}, "
            f"min SDR vs TF={min(sdr) if sdr else float('inf'):.1f} dB, ORT {ort_s:.1f}s")
        if max_abs > SELF_CHECK_MAX_ABS or (sdr and min(sdr) < SELF_CHECK_MIN_SDR_DB):
            raise SystemExit("self-check: ONNX output differs from the TF graph by more than the allowed tolerance")
    tf_sess.close()


# --------------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=int, choices=sorted(CHECKPOINTS), required=True, help="number of output sources")
    ap.add_argument("--out", required=True, help="output .onnx file")
    ap.add_argument("--ckpt-dir", help="directory with inference.meta and the model.ckpt-* files (download target with --download)")
    ap.add_argument("--download", action="store_true", help="fetch the checkpoint from the upstream bucket (sha256-verified)")
    ap.add_argument("--opset", type=int, default=DEFAULT_OPSET, help=f"ONNX opset (default {DEFAULT_OPSET}, the maximum tf2onnx 1.16 supports; 13 was verified too)")
    args = ap.parse_args()
    if not args.download and not args.ckpt_dir:
        ap.error("give --ckpt-dir DIR and/or --download")

    tmp_dir = None
    ckpt_dir = args.ckpt_dir
    if args.download:
        if not ckpt_dir:
            tmp_dir = ckpt_dir = tempfile.mkdtemp(prefix="mixit_ckpt_")
        download_checkpoint(args.sources, ckpt_dir)
    try:
        verify_checkpoint(args.sources, ckpt_dir)
        spec = CHECKPOINTS[args.sources]
        t0 = time.time()
        log(f"freezing {spec['subdir']}/{spec['prefix']}")
        frozen = freeze_checkpoint(ckpt_dir, spec["prefix"])
        log(f"frozen graph: {len(frozen.node)} nodes")
        fused, n_fused = fuse_dilated_depthwise(frozen)
        log(f"fused {n_fused} dilated depthwise convolutions -> {len(fused.node)} nodes")
        model = finalize_model(to_onnx(fused, args.opset), args.sources, args.opset)
        import onnx

        onnx.checker.check_model(model)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        onnx.save(model, args.out)
        log(f"wrote {args.out}: {os.path.getsize(args.out) / 1e6:.1f} MB, {len(model.graph.node)} nodes, opset {args.opset}")
        self_check(args.out, frozen, args.sources)
        log(f"done in {time.time() - t0:.0f}s")
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
