# Lecture Video Generator

Generates a full lecture video from a slide script and a PDF of slides.

- **script.json** — narration text per slide
- **slides.pdf** — one page per slide

The pipeline clones a reference voice with VoxCPM2, converts each slide's text to speech, renders each slide image with its audio into a video clip, and concatenates all clips into a single `lecture.mp4`.

## Repository Files

| File | Description |
|------|-------------|
| `lecture_audio_plain.ipynb` | Original Colab notebook (manual cell-by-cell) |
| `lvm_colab.ipynb` | Colab-specific notebook with Drive caching and tunnel dropdown |
| `lvm.ipynb` | Generic Jupyter notebook (works on Lightning AI, Kaggle, local) |
| `app.py` | Standalone Gradio application — platform agnostic (standard voxcpm backend) |
| `app_nanovllm.py` | High-performance Gradio app using **Nano-vLLM VoxCPM** backend |
| `requirements.txt` | Dependencies for standard backend |
| `requirements-nanovllm.txt` | Dependencies for Nano-vLLM backend |

## Quick Start (Lightning AI / Generic Jupyter)

1. Open a terminal in your studio and clone the repo:
   ```bash
   git clone https://github.com/Mwimwii/lvm.git
   git clone https://github.com/Mwimwii/vo.git
   cd lvm
   ```
2. Open `lvm.ipynb` and run both cells.
   - Cell 1 installs dependencies.
   - Cell 2 launches the Gradio UI with `--share`.
3. Or run directly from the terminal:
   ```bash
   cd lvm
   pip install -r requirements.txt
   python app.py --share
   ```

## Remote Batch Processing via R2 (Client / Server)

For heavy workloads, use the **client/server** architecture. Your local machine runs the Gradio client. The actual generation runs as a one-shot Lightning AI Docker Job on an **L40S** GPU, with all data flowing through Cloudflare R2.

### Build the Docker Image (one-time)

The Docker image pre-bakes CUDA, Python, FFmpeg, all pip deps, and the VoxCPM2 model (~5 GB). This eliminates cold starts entirely.

**Option A: Build locally (Windows/Mac/Linux)**

```bash
cd lvm

# Build (takes ~15-20 min, image ~10 GB)
docker build -t mpnyirongo/lvm-processor:latest .

# Login to Docker Hub (create account at hub.docker.com if needed)
docker login

# Push
docker push mpnyirongo/lvm-processor:latest
```

**Option B: Build on your Lightning AI Studio**

If you don't have Docker locally, build it right on your existing Studio:

```bash
# In your Studio terminal
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
# Re-login, then:
cd ~/lvm
docker build -t mpnyirongo/lvm-processor:latest .
docker login
docker push mpnyirongo/lvm-processor:latest
```

### Setup

1. Create a Cloudflare R2 bucket and generate an S3-compatible API token.
2. Copy `lvm/.env.example` to `lvm/.env` and fill in your credentials:
   ```bash
   cd lvm
   cp .env.example .env
   # Edit .env with your R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL, R2_BUCKET_NAME
   ```
3. Install client dependencies:
   ```bash
   pip install -r requirements-client.txt
   ```

### Run the Client

```bash
python app_client.py
```

The client UI runs on `http://localhost:7860`. It will:

1. Upload your `script.json`, `slides.pdf`, and optional reference voice to R2 under `jobs/{job_id}/inputs/`.
2. Submit a one-shot L40S Docker Job via `Job.run()` using the pre-built `ghcr.io/Mwimwii/lvm-processor` image.
3. The container downloads inputs from R2, runs the nano-vllm VoxCPM pipeline, uploads outputs back to R2, then auto-terminates.
4. The client polls R2 every 30 seconds. When it sees the `.mp4`, it downloads it locally.

### Resume after Disconnect

If your internet drops, just re-open the client, enter your Job ID (e.g. `job_a1b2c3d4`), and click **Check / Resume**. The client fetches outputs directly from R2 — no state is stored locally.

### Files

| File | Description |
|------|-------------|
| `app_client.py` | Local Gradio client. Uploads to R2, submits Lightning Docker Job, polls for outputs. |
| `app_batch_processor.py` | Headless nano-vllm pipeline script executed inside the L40S container. |
| `app_r2_bootstrap.py` | Bootstrap script that downloads inputs from R2 and runs the batch processor. |
| `Dockerfile` | Pre-bakes CUDA + Python + FFmpeg + deps + VoxCPM2 model into a Docker image. |
| `requirements-client.txt` | Client-only deps (boto3, python-dotenv, gradio, lightning-sdk). |

## Quick Start (Google Colab)

1. Upload `lvm_colab.ipynb` to Google Colab.
2. Run the **Installation** cell.
   - Check **Mount Google Drive** to cache the ~5GB VoxCPM2 model between sessions. The first download takes time, but subsequent sessions load instantly from Drive.
   - It clones this repo, the default voice repo, and installs dependencies.
3. Run the **Run UI** cell.
   - Choose a tunnel (Gradio, Ngrok, Cloudflare, LocalTunnel, or Horizon).
   - Fill in credentials if needed (Ngrok authtoken, Horizon ID).
4. Open the public URL that appears.
5. In the web UI:
   - Optionally upload a custom reference voice (`.wav` + transcript).
   - Upload `script.json` and `slides.pdf`.
   - Adjust CFG, inference steps, and video height if desired.
   - Click **Generate Lecture Video**.
6. Download `lecture.mp4` and `lecture.wav` from the output section.

## Running Locally (Standard Backend)

Requires Python 3.10+, CUDA-capable GPU, and FFmpeg.

```bash
git clone https://github.com/Mwimwii/lvm.git
git clone https://github.com/Mwimwii/vo.git
cd lvm
pip install -r requirements.txt
python app.py --share
```

Without `--share`, the app serves on `0.0.0.0:9999` for use with your own tunnel.

You can also point to a custom reference voice via environment variable:
```bash
REF_VOICE_PATH=/path/to/speaker.wav python app.py --share
```

## Running with Nano-vLLM Backend (High Performance)

For significantly faster inference and concurrent request support, use the **Nano-vLLM VoxCPM** backend. This is recommended for H100/A100 GPUs and high-throughput scenarios.

Prerequisites:
- Linux + NVIDIA GPU with CUDA
- Model weights in **`.safetensors`** format (not `.pt`)
- `flash-attn` installed

```bash
git clone https://github.com/Mwimwii/lvm.git
cd lvm
pip install -r requirements-nanovllm.txt
python app_nanovllm.py --share --model openbmb/VoxCPM2
```

Options:
- `--model` — HuggingFace repo id or local path to VoxCPM2 weights
- `--gpu-mem 0.95` — GPU memory utilization fraction (default 0.95)

**Reference voice cloning:** The nanovllm backend now supports full voice cloning. When you upload a reference `.wav` + transcript in the UI and click **Save Reference Voice**, the app encodes the audio into `prompt_latents` via the server's AudioVAE. These latents are passed to every `generate()` call, cloning the reference voice for the entire lecture.

## Input Format

**script.json**

```json
{
  "1": ["First sentence for slide 1.", "Second sentence for slide 1."],
  "2": ["Sentence for slide 2."]
}
```

Keys are 1-based slide numbers matching the PDF page order.

**slides.pdf**

One page per slide. Pages are exported to PNG at 2x scale, then resized to the chosen video height during FFmpeg encoding.

## Pipeline

```
script.json  ->  VoxCPM2 TTS  ->  slides_audio/slide_001.wav
slides.pdf   ->  PyMuPDF       ->  slides_images/slide_001.png
                                    |
                                    v
                          FFmpeg (image + audio -> clip)
                                    |
                                    v
                          FFmpeg concat -> lecture.mp4
```

## Caching Models on Google Drive

The VoxCPM2 model is approximately 5GB and downloads from HuggingFace and ModelScope. Colab VMs are ephemeral, so without caching the model re-downloads every session.

To avoid this, check **Mount Google Drive** in the Installation cell. This sets:

- `HF_HOME` -> `Drive/MyDrive/ai_cache/huggingface`
- `MODELSCOPE_CACHE` -> `Drive/MyDrive/ai_cache/modelscope`

First run downloads to Drive. Later runs load from Drive in seconds.

## Troubleshooting

**`AssertionError` in `torch._inductor.cudagraph_trees`**

This happens when VoxCPM's internal `torch.compile` runs inside a Gradio worker thread. The fix disables compilation entirely. If you hit this error:

1. Restart the runtime (**Runtime > Restart runtime**).
2. Re-run both cells. The `TORCH_COMPILE_DISABLE=1` environment variable prevents compilation, avoiding the thread-local state issue.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| CFG | 2.7 | Voice guidance scale. Higher values stick closer to the reference voice. |
| TTS Inference Steps | 10 | Denoising steps. More steps improve quality but slow generation. |
| Video Height | 720 | Output height in pixels. Width scales automatically. |
| Clear Assets | true | Wipe previous run's audio, images, and clips before generating. |

## Reference Voice

The app searches for the default reference voice in this order:

1. `REF_VOICE_PATH` environment variable
2. `lvm/vo/speaker.wav` (repo-subfolder)
3. `../vo/speaker.wav` (sibling repo)
4. `/content/vo/speaker.wav` (legacy Colab)

To use a custom voice, either:
- Clone the `vo` repo next to or inside the `lvm` folder, **or**
- Set `REF_VOICE_PATH=/path/to/speaker.wav`, **or**
- Upload a `.wav` + transcript in the UI and click **Save Reference Voice**.
