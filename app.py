#!/usr/bin/env python3
"""
Lecture Video Generator — Gradio UI
Run inside the lvm/ repo on Colab (or anywhere with CUDA + FFmpeg).

Usage:
    python app.py              # serve on 0.0.0.0:9999 (for Ngrok/Cloudflare/etc.)
    python app.py --share      # serve with Gradio share tunnel
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

# Disable torch.compile entirely. VoxCPM uses it internally, but it breaks
# in Gradio worker threads on Colab T4 because torch inductor CUDA graphs
# rely on thread-local state that is absent outside the main thread.
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import numpy as np
import soundfile as sf
import fitz  # PyMuPDF
import torch
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
from voxcpm import VoxCPM

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48_000
GAP_BETWEEN_SEGMENTS = 0.3   # seconds of silence between sentences in a slide
GAP_BETWEEN_SLIDES = 1.0     # seconds of silence between slides
VIDEO_FPS = 10
AUDIO_BITRATE = "192k"

SLIDES_AUDIO_DIR = "slides_audio"
SLIDES_IMAGES_DIR = "slides_images"
SLIDES_VIDEO_DIR = "slides_video"
AUDIO_OUTPUT = "lecture.wav"
VIDEO_OUTPUT = "lecture.mp4"

DEFAULT_REF_WAV = "/content/vo/speaker.wav"
DEFAULT_REF_TEXT = (
    "A string is defined as a sequence of characters, and it is important to "
    "remember that these are case sensitive. Strings can contain anything from "
    "standard alphabet letters, the special symbols, spaces and numerical digits."
)

# ── Global state ──────────────────────────────────────────────────────────────
model: VoxCPM | None = None
cached_ref_wav: str | None = None
cached_ref_text: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ensure_dirs() -> None:
    for d in (SLIDES_AUDIO_DIR, SLIDES_IMAGES_DIR, SLIDES_VIDEO_DIR):
        os.makedirs(d, exist_ok=True)


def _clear_dir(directory: str, extension: str) -> None:
    if os.path.isdir(directory):
        for f in os.listdir(directory):
            if f.endswith(extension):
                os.remove(os.path.join(directory, f))


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def _generate_segment(text: str, cfg: float, timesteps: int) -> np.ndarray:
    assert model is not None
    assert cached_ref_wav is not None
    assert cached_ref_text is not None
    return model.generate(
        text=text,
        prompt_wav_path=cached_ref_wav,
        prompt_text=cached_ref_text,
        cfg_value=cfg,
        inference_timesteps=timesteps,
        normalize=True,
        denoise=True,
        retry_badcase=True,
        retry_badcase_max_times=3,
        retry_badcase_ratio_threshold=6.0,
    )


def _pdf_to_images(pdf_path: str) -> int:
    _clear_dir(SLIDES_IMAGES_DIR, ".png")
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        slide_num = page_num + 1
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(SLIDES_IMAGES_DIR, f"slide_{str(slide_num).zfill(3)}.png")
        pix.save(out_path)
    count = len(doc)
    doc.close()
    return count


def _process_slide_audio(
    slide_num: str,
    segments: list[str],
    cfg: float,
    timesteps: int,
    progress: gr.Progress,
    total_segments: int,
    done_counter: list[int],
) -> tuple[str, np.ndarray]:
    wav_path = os.path.join(SLIDES_AUDIO_DIR, f"slide_{slide_num.zfill(3)}.wav")
    chunks: list[np.ndarray] = []
    seg_silence = _silence(GAP_BETWEEN_SEGMENTS)

    for idx, text in enumerate(segments):
        wav = _generate_segment(text, cfg, timesteps)
        chunks.append(wav)
        if idx < len(segments) - 1:
            chunks.append(seg_silence)
        done_counter[0] += 1
        progress(
            done_counter[0] / total_segments,
            desc=f"Slide {slide_num}  segment {idx + 1}/{len(segments)}",
        )

    slide_wav = np.concatenate(chunks)
    sf.write(wav_path, slide_wav, SAMPLE_RATE)
    return slide_num, slide_wav


def _make_slide_clip_worker(args: tuple) -> tuple[int, str] | None:
    num, image_path, audio_path, output_path, video_height = args
    if os.path.exists(output_path):
        os.remove(output_path)
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", image_path, "-i", audio_path,
        "-c:v", "libx264", "-tune", "stillimage",
        "-vf", f"scale=-2:{video_height},format=yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-r", str(VIDEO_FPS), "-shortest", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return num, output_path
    print(f"[FFmpeg error slide {num}] {result.stderr[-300:]}")
    return None


# ── Reference voice handlers ──────────────────────────────────────────────────
def save_reference(audio_file: str | None, text: str) -> str:
    global cached_ref_wav, cached_ref_text
    if audio_file is None:
        return "⚠️ Please upload a reference audio file."

    dest = "/content/ref_voice.wav"
    shutil.copy(audio_file, dest)
    cached_ref_wav = dest
    cached_ref_text = text.strip() if text and text.strip() else DEFAULT_REF_TEXT
    size_mb = os.path.getsize(dest) / 1024 / 1024
    return (
        f"✅ Reference saved: `{dest}` ({size_mb:.1f} MB)\n\n"
        f"**Transcript:** {cached_ref_text[:120]}{'…' if len(cached_ref_text) > 120 else ''}"
    )


def get_ref_status() -> str:
    if cached_ref_wav and os.path.exists(cached_ref_wav):
        size_mb = os.path.getsize(cached_ref_wav) / 1024 / 1024
        return (
            f"📎 Current reference: `{cached_ref_wav}` ({size_mb:.1f} MB)\n\n"
            f"**Transcript:** {cached_ref_text[:120]}{'…' if len(cached_ref_text) > 120 else ''}"
        )
    if os.path.exists(DEFAULT_REF_WAV):
        return f"📎 Using default reference: `{DEFAULT_REF_WAV}` (run *Save Reference* to override)."
    return "⚠️ No reference voice found. Upload one below or clone the `vo` repo first."


# ── Main generation handler ───────────────────────────────────────────────────
def generate_lecture(
    script_file: str | None,
    pdf_file: str | None,
    cfg: float,
    timesteps: int,
    video_height: int,
    clear_assets: bool,
    progress: gr.Progress = gr.Progress(),
):
    global cached_ref_wav, cached_ref_text

    # Fallback to default reference
    if cached_ref_wav is None or not os.path.exists(cached_ref_wav):
        if os.path.exists(DEFAULT_REF_WAV):
            cached_ref_wav = DEFAULT_REF_WAV
            cached_ref_text = DEFAULT_REF_TEXT
        else:
            yield None, None, "❌ Reference voice not available. Upload a custom one or clone the `vo` repo."
            return

    if script_file is None:
        yield None, None, "❌ Please upload **script.json**."
        return
    if pdf_file is None:
        yield None, None, "❌ Please upload **slides.pdf**."
        return

    _ensure_dirs()

    if clear_assets:
        _clear_dir(SLIDES_AUDIO_DIR, ".wav")
        _clear_dir(SLIDES_IMAGES_DIR, ".png")
        _clear_dir(SLIDES_VIDEO_DIR, ".mp4")
        for f in (AUDIO_OUTPUT, VIDEO_OUTPUT):
            if os.path.exists(f):
                os.remove(f)

    yield None, None, "📄 Loading script & converting PDF to images..."

    script_path = "/content/script.json"
    pdf_path = "/content/slides.pdf"
    shutil.copy(script_file, script_path)
    shutil.copy(pdf_file, pdf_path)

    with open(script_path) as f:
        script: dict[str, list[str]] = json.load(f)
    slide_keys = sorted(script.keys(), key=lambda k: int(k))
    total_slides = len(slide_keys)
    total_segments = sum(len(script[k]) for k in slide_keys)

    num_pages = _pdf_to_images(pdf_path)

    yield (
        None,
        None,
        f"🖼️ PDF: **{num_pages}** pages | Script: **{total_slides}** slides | "
        f"Segments: **{total_segments}**\n\n🔊 Generating audio...",
    )

    # ── Audio generation ──
    _clear_dir(SLIDES_AUDIO_DIR, ".wav")
    done_counter = [0]
    results: list[tuple[str, np.ndarray]] = []
    for slide_num in slide_keys:
        segments = script[slide_num]
        _, slide_wav = _process_slide_audio(slide_num, segments, cfg, timesteps, progress, total_segments, done_counter)
        results.append((slide_num, slide_wav))

    # Concatenate full audio
    results.sort(key=lambda x: int(x[0]))
    all_audio: list[np.ndarray] = []
    slide_silence = _silence(GAP_BETWEEN_SLIDES)
    for _, slide_wav in results:
        all_audio.extend([slide_wav, slide_silence])
    full_audio = np.concatenate(all_audio)
    sf.write(AUDIO_OUTPUT, full_audio, SAMPLE_RATE)

    yield None, None, f"🎬 Audio complete. Assembling **{len(results)}** video clips with FFmpeg..."

    # ── Video clips ──
    available_images = {
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(SLIDES_IMAGES_DIR)
        if f.endswith(".png")
    }
    available_audio = {
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(SLIDES_AUDIO_DIR)
        if f.endswith(".wav")
    }
    matched_slides = sorted(available_images & available_audio)

    tasks: list[tuple] = []
    for num in matched_slides:
        key = str(num).zfill(3)
        tasks.append((
            num,
            os.path.join(SLIDES_IMAGES_DIR, f"slide_{key}.png"),
            os.path.join(SLIDES_AUDIO_DIR, f"slide_{key}.wav"),
            os.path.join(SLIDES_VIDEO_DIR, f"slide_{key}.mp4"),
            video_height,
        ))

    _clear_dir(SLIDES_VIDEO_DIR, ".mp4")
    with ThreadPoolExecutor(max_workers=4) as executor:
        clip_results = list(executor.map(_make_slide_clip_worker, tasks))

    successful = sorted([r for r in clip_results if r is not None], key=lambda x: x[0])
    clip_paths = [path for _, path in successful]

    yield None, None, f"🎞️ **{len(clip_paths)}** clips ready. Concatenating final lecture.mp4..."

    # ── Final concat ──
    concat_list = "/content/concat_list.txt"
    with open(concat_list, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{clip}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list, "-c", "copy", VIDEO_OUTPUT,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(concat_list)

    if result.returncode != 0:
        yield None, None, f"❌ FFmpeg concat failed:\n```\n{result.stderr[-400:]}\n```"
        return

    size_mb = os.path.getsize(VIDEO_OUTPUT) / 1024 / 1024
    yield (
        VIDEO_OUTPUT,
        AUDIO_OUTPUT,
        f"✅ **Done!**\n\n"
        f"📹 `{VIDEO_OUTPUT}` — **{size_mb:.1f} MB**\n"
        f"🔊 `{AUDIO_OUTPUT}` — full audio track",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ═══════════════════════════════════════════════════════════════════════════════
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Lecture Video Generator") as demo:
        gr.Markdown("# 🎓 Lecture Video Generator")
        gr.Markdown(
            "Generate a full lecture video from a **script.json** and **slides.pdf** "
            "using VoxCPM2 voice cloning."
        )

        # ── Reference Voice Section ──
        with gr.Accordion("🔊 Reference Voice Setup", open=True):
            ref_status = gr.Markdown(value=get_ref_status())
            with gr.Row():
                ref_audio = gr.Audio(
                    label="Upload Reference Audio (.wav)",
                    type="filepath",
                    sources=["upload"],
                )
                ref_text = gr.Textbox(
                    label="Reference Transcript",
                    lines=3,
                    placeholder="Paste the exact spoken text of the reference audio...",
                )
            save_ref_btn = gr.Button("💾 Save Reference Voice", variant="secondary")
            save_ref_btn.click(
                fn=save_reference,
                inputs=[ref_audio, ref_text],
                outputs=ref_status,
            )

        # ── Inputs ──
        gr.Markdown("## 📤 Lecture Inputs")
        with gr.Row():
            script_input = gr.File(label="script.json", file_types=[".json"])
            pdf_input = gr.File(label="slides.pdf", file_types=[".pdf"])

        # ── Settings ──
        gr.Markdown("## ⚙️ Generation Settings")
        with gr.Row():
            cfg_slider = gr.Slider(
                minimum=1.0, maximum=5.0, value=2.7, step=0.1,
                label="CFG (voice fidelity)",
                info="Higher = closer to reference voice",
            )
            timesteps_slider = gr.Slider(
                minimum=5, maximum=25, value=10, step=1,
                label="TTS Inference Steps",
                info="Higher = better quality, slower",
            )
            video_height = gr.Dropdown(
                choices=[480, 720, 1080],
                value=720,
                label="Video Height (px)",
            )
        clear_checkbox = gr.Checkbox(
            value=True,
            label="Clear previous assets before generation",
        )

        generate_btn = gr.Button("🚀 Generate Lecture Video", variant="primary")

        # ── Outputs ──
        gr.Markdown("## 📥 Outputs")
        with gr.Row():
            output_video = gr.Video(label="lecture.mp4")
            output_audio = gr.Audio(label="lecture.wav", type="filepath")
        status_md = gr.Markdown("*Status will appear here...*")

        generate_btn.click(
            fn=generate_lecture,
            inputs=[
                script_input,
                pdf_input,
                cfg_slider,
                timesteps_slider,
                video_height,
                clear_checkbox,
            ],
            outputs=[output_video, output_audio, status_md],
            show_progress="minimal",
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Lecture Video Generator UI")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    print("Loading VoxCPM2...")
    global model
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2")
    print("Model ready.")

    demo = build_ui()

    if args.share:
        print("Launching with Gradio share link...")
        demo.launch(share=True)
    else:
        print("Launching on 0.0.0.0:9999 (use Ngrok/Cloudflare/LocalTunnel/Horizon to expose)")
        demo.launch(server_name="0.0.0.0", server_port=9999)


if __name__ == "__main__":
    main()
