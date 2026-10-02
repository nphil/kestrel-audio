# Third-party notices

Kestrel Audio's own code is MIT-licensed (see `LICENSE`). The container image also carries the components below.

## Models

**Google Perch v2** (bird vocalization classifier, 14,795 classes): Copyright Google LLC, Apache License 2.0
(https://www.apache.org/licenses/LICENSE-2.0). Weights: https://www.kaggle.com/models/google/bird-vocalization-classifier .
The ONNX build used here (`perch_v2_no_dft_fp32.onnx`, sha256 `4dcf71c18a147198545944bb5149697e89e3ad2e16637fa8f0edf6d13035a017`)
and its label list are the conversion published by Tomi P. Hakala at https://huggingface.co/tphakala/Perch-v2-Models ; the
image downloads them at build time and does not modify them. Paper: B. van Merriënboer et al., "Perch 2.0: The Bittern Lesson
for Bioacoustics", 2025.

**Bird MixIT sound-separation models (4-source and 8-source)**: Copyright Google LLC, licensed under the Apache License 2.0
(https://www.apache.org/licenses/LICENSE-2.0). Source: https://github.com/google-research/sound-separation/tree/master/models/bird_mixit ,
checkpoints `gs://gresearch/sound_separation/bird_mixit_model_checkpoints/` (`output_sources4/model.ckpt-3223090`,
`output_sources8/model.ckpt-2178900`). **Modified:** Kestrel Audio ships these weights converted to ONNX by
`tools/convert_mixit.py` (graph rewritten for onnxruntime, numerics checked against the TensorFlow original); the conversion
changes the file format, not the trained weights. Please cite S. Wisdom et al., "Unsupervised Sound Separation Using Mixture
Invariant Training", NeurIPS 2020 (https://arxiv.org/abs/2006.12701), and T. Denton, S. Wisdom, J. R. Hershey, "Improving Bird
Classification with Unsupervised Sound Separation", ICASSP 2022 (https://arxiv.org/abs/2110.03209).

## Runtime

* **ONNX Runtime** (`onnxruntime-gpu` 1.26.0), MIT, Microsoft.
* **NVIDIA CUDA 12.6 runtime, cuBLAS and cuDNN 9** from the `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` base image, under the
  NVIDIA Deep Learning Container License (`/NGC-DL-CONTAINER-LICENSE` inside the image) and the CUDA EULA
  (https://docs.nvidia.com/cuda/eula/index.html).
* **FFmpeg** (Ubuntu package), LGPL/GPL; used as a separate program to decode clips and encode AAC.
* **NumPy**, **SciPy** (BSD-3-Clause), **pyloudnorm** (MIT), **FastAPI**, **Starlette** (MIT / BSD-3-Clause), **Uvicorn**
  (BSD-3-Clause), **nvidia-ml-py** (BSD-3-Clause).
* **tini** (MIT), Ubuntu 24.04 packages.

## Build stage only (not in the running image)

TensorFlow 2.12 (Apache-2.0), tf2onnx 1.16 (Apache-2.0), ONNX 1.16 (Apache-2.0).

## Status page

Fonts: **Manrope** (SIL Open Font License 1.1, https://github.com/sharanda/manrope), Latin subset, inlined. Design tokens:
the **Lucent** design language and the **Rosé Pine** palette (MIT, https://rosepinetheme.com).
