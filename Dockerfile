# ── LVM Batch Processor Image ─────────────────────────────────────────────
# CUDA 12.4 + Python 3.10 + nano-vllm-voxcpm + FFmpeg
# Pre-bakes all deps and the VoxCPM2 model to eliminate cold starts.
# ─────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="8.9"

WORKDIR /app

# ── System deps ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1-dev \
    libgl1-mesa-glx \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────────────────────
# Install torch first (already in base image, but ensure it's in env)
RUN pip install --no-cache-dir ninja packaging wheel

# Install flash-attn with torch available in build environment
RUN pip install --no-cache-dir flash-attn --no-build-isolation \
    && python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__} installed OK')"

# Install remaining deps
RUN pip install --no-cache-dir \
    git+https://github.com/a710128/nanovllm-voxcpm.git \
    soundfile \
    numpy \
    pymupdf \
    boto3 \
    botocore

# ── Pre-download VoxCPM2 model (~5 GB) via huggingface_hub (no GPU needed) ──
RUN python -c "\
from huggingface_hub import snapshot_download; \
print('Downloading VoxCPM2 model...'); \
snapshot_download('openbmb/VoxCPM2', local_dir='/app/hf_cache/VoxCPM2'); \
print('Done.') \
"

# ── Embed batch processor + bootstrap ────────────────────────────────────
COPY app_batch_processor.py /app/batch.py
COPY app_r2_bootstrap.py /app/bootstrap.py

# ── Entrypoint ───────────────────────────────────────────────────────────
# Lightning Jobs default entrypoint is "sh -c", so we set CMD instead.
# The client will pass: command="python /app/bootstrap.py"
CMD ["python", "/app/bootstrap.py"]
