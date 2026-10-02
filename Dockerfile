# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------------------------------------------------
# Stage 1 - the models.
#  * Google Perch v2 (ONNX, FP32) comes from the Hugging Face mirror that BirdNET-Go's author maintains.
#  * Google's bird MixIT checkpoints are TensorFlow 1 graphs: they are converted to ONNX here. TensorFlow exists only in
#    this stage; the runtime image below runs everything through onnxruntime.
FROM python:3.11-slim-bookworm AS models
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY tools/requirements-convert.txt tools/convert_mixit.py tools/
RUN pip install --no-cache-dir -r tools/requirements-convert.txt
RUN mkdir -p /models \
 && python tools/convert_mixit.py --sources 4 --download --out /models/mixit4.onnx \
 && python tools/convert_mixit.py --sources 8 --download --out /models/mixit8.onnx
ARG PERCH_BASE=https://huggingface.co/tphakala/Perch-v2-Models/resolve/main/full
RUN curl -fsSL --retry 5 --retry-delay 5 -o /models/perch_v2_no_dft_fp32.onnx "$PERCH_BASE/perch_v2_no_dft_fp32.onnx" \
 && echo "4dcf71c18a147198545944bb5149697e89e3ad2e16637fa8f0edf6d13035a017  /models/perch_v2_no_dft_fp32.onnx" | sha256sum -c - \
 && curl -fsSL --retry 5 --retry-delay 5 -o /models/perch_v2_labels.txt "$PERCH_BASE/perch_v2_labels.txt" \
 && echo "e4d5c0397d8fb08bf90c6b13a34810af53504faad927e472fcc567793c9de057  /models/perch_v2_labels.txt" | sha256sum -c -

# ---------------------------------------------------------------------------------------------------------------------
# Stage 2 - the service. CUDA 12 + cuDNN 9 runtime libraries (onnxruntime-gpu 1.26 is the last build for CUDA 12; newer ones
# target CUDA 13, which dropped Pascal cards such as the Tesla P40).
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg curl tini ca-certificates tzdata \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY --from=models /models /models
WORKDIR /app
COPY kestrel_audio ./kestrel_audio
COPY assets/icon-512.png assets/favicon-32.png assets/favicon.ico ./assets/

ARG VERSION=0.0.0-dev
ENV KESTREL_AUDIO_VERSION=${VERSION} \
    KESTREL_AUDIO_MODELS=/models \
    KESTREL_AUDIO_DATA=/data \
    KESTREL_AUDIO_ASSETS=/app/assets \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
LABEL org.opencontainers.image.title="Kestrel Audio" \
      org.opencontainers.image.description="Makes Kestrel bird and animal sound previews audible, and cleaner only when proven safe" \
      org.opencontainers.image.source="https://github.com/nphil/kestrel-audio" \
      org.opencontainers.image.licenses="MIT"

VOLUME /data
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${KESTREL_AUDIO_PORT:-8787}/healthz" || exit 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "kestrel_audio"]
