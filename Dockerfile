# ── LVM Batch Processor Image ─────────────────────────────────────────────
# CUDA 12.4 + Python 3.10 + nano-vllm-voxcpm + FFmpeg
# Pre-bakes all deps and the VoxCPM2 model to eliminate cold starts.
# ─────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

# ── System deps ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    ffmpeg \
    libsndfile1-dev \
    libgl1-mesa-glx \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Python deps ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir \
    nano-vllm-voxcpm \
    soundfile \
    numpy \
    pymupdf \
    boto3 \
    botocore

# ── Pre-download VoxCPM2 model (~5 GB) ───────────────────────────────────
RUN python -c "\
from nanovllm_voxcpm import VoxCPM; \
print('Downloading VoxCPM2 model...'); \
server = VoxCPM.from_pretrained('openbmb/VoxCPM2', devices=[0]); \
print('Model cached successfully.') \
"

# ── Embed batch processor + bootstrap ────────────────────────────────────
COPY app_batch_processor.py /app/batch.py
COPY app_r2_bootstrap.py /app/bootstrap.py

# ── Entrypoint ───────────────────────────────────────────────────────────
# Lightning Jobs default entrypoint is "sh -c", so we set CMD instead.
# The client will pass: command="python /app/bootstrap.py"
CMD ["python", "/app/bootstrap.py"]
