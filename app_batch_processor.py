#!/usr/bin/env python3
"""
Lecture Video Generator — Batch Processor
Runs inside a Lightning AI Job container (H100).
Reads inputs from --work-dir, writes outputs to --output-dir.

Usage:
    python app_batch_processor.py \
        --job-id job_abc12345 \
        --work-dir /teamspace/jobs/job_abc12345 \
        --output-dir /teamspace/drive/jobs/job_abc12345/output
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
import fitz
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from nanovllm_voxcpm import VoxCPM

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48_000
GAP_BETWEEN_SEGMENTS = 0.3
GAP_BETWEEN_SLIDES = 1.0
VIDEO_FPS = 10
AUDIO_BITRATE = "192k"

# ── Server singleton ──────────────────────────────────────────────────────────
server = None

# ── Cached reference voice state ─────────────────────────────────────────────-
cached_ref_latents: bytes | None = None
cached_ref_text: str | None = None


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def _encode_ref_audio(wav_path: str) -> bytes:
    assert server is not None
    with open(wav_path, "rb") as f:
        return server.encode_latents(f.read(), "wav")


def _generate_segment(text: str, cfg: float) -> np.ndarray:
    chunks = []
    kwargs = {"target_text": text, "cfg_value": cfg}
    if cached_ref_latents is not None and cached_ref_text:
        kwargs["prompt_latents"] = cached_ref_latents
        kwargs["prompt_text"] = cached_ref_text
    for chunk in server.generate(**kwargs):
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0)


def _pdf_to_images(pdf_path: str, img_dir: str, max_pages: int | None = None) -> int:
    os.makedirs(img_dir, exist_ok=True)
    for f in os.listdir(img_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(img_dir, f))
    doc = fitz.open(pdf_path)
    total = len(doc)
    end = min(total, max_pages) if max_pages else total
    for page_num in range(end):
        slide_num = page_num + 1
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(img_dir, f"slide_{str(slide_num).zfill(3)}.png")
        pix.save(out_path)
    doc.close()
    return end


def _process_slide_audio(
    slide_num: str,
    segments: list[str],
    cfg: float,
    audio_dir: str,
    total_segments: int,
    done_counter: list[int],
) -> tuple[str, np.ndarray]:
    wav_path = os.path.join(audio_dir, f"slide_{slide_num.zfill(3)}.wav")
    chunks: list[np.ndarray] = []
    seg_silence = _silence(GAP_BETWEEN_SEGMENTS)

    for idx, text in enumerate(segments):
        wav = _generate_segment(text, cfg)
        chunks.append(wav)
        if idx < len(segments) - 1:
            chunks.append(seg_silence)
        done_counter[0] += 1
        print(f"[Audio] Slide {slide_num} segment {idx + 1}/{len(segments)} ({done_counter[0]}/{total_segments})")

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


def run_job(job_id: str, work_dir: str, output_dir: str) -> None:
    global cached_ref_latents, cached_ref_text

    script_path = os.path.join(work_dir, "script.json")
    pdf_path = os.path.join(work_dir, "slides.pdf")
    manifest_path = os.path.join(work_dir, "manifest.json")

    if not os.path.exists(script_path) or not os.path.exists(pdf_path):
        print("❌ Missing script.json or slides.pdf")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    cfg = manifest["cfg"]
    video_height = manifest["video_height"]
    clear_assets = manifest["clear_assets"]
    preview_start = manifest.get("preview_start")
    preview_end = manifest.get("preview_end")
    ref_text_from_manifest = manifest.get("ref_text", "")
    output_name = manifest.get("output_name")

    audio_dir = os.path.join(work_dir, "slides_audio")
    img_dir = os.path.join(work_dir, "slides_images")
    video_dir = os.path.join(work_dir, "slides_video")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── Reference voice ──
    custom_ref = os.path.join(work_dir, "ref_voice.wav")
    if os.path.exists(custom_ref):
        ref_wav = custom_ref
        ref_text = ref_text_from_manifest or "This is the reference voice speaking."
        print("Using uploaded reference voice.")
    else:
        defaults = [
            "/teamspace/studios/this_studio/vo/speaker.wav",
            "/teamspace/studios/this_studio/lvm/vo/speaker.wav",
        ]
        ref_wav = None
        ref_text = ref_text_from_manifest or (
            "A string is defined as a sequence of characters, and it is important to "
            "remember that these are case sensitive."
        )
        for p in defaults:
            if os.path.exists(p):
                ref_wav = p
                print(f"Using default reference: {p}")
                break

    if ref_wav is None or not os.path.exists(ref_wav):
        print("❌ No reference voice found.")
        sys.exit(1)

    cached_ref_latents = _encode_ref_audio(ref_wav)
    print(f"Latents cached: {len(cached_ref_latents)} bytes")

    # ── Load script ──
    with open(script_path) as f:
        script: dict[str, list[str]] = json.load(f)
    all_slide_keys = sorted(script.keys(), key=lambda k: int(k))

    is_preview = preview_start is not None and preview_end is not None
    if is_preview:
        slide_keys = [k for k in all_slide_keys if preview_start <= int(k) <= preview_end]
        if not slide_keys:
            print(f"⚠️ No slides in range {preview_start}–{preview_end}")
            sys.exit(1)
    else:
        slide_keys = all_slide_keys

    total_slides = len(slide_keys)
    total_segments = sum(len(script[k]) for k in slide_keys)
    print(f"Job {job_id}: {total_slides} slides, {total_segments} segments.")

    if clear_assets:
        for d, ext in [(audio_dir, ".wav"), (img_dir, ".png"), (video_dir, ".mp4")]:
            for f in os.listdir(d):
                if f.endswith(ext):
                    os.remove(os.path.join(d, f))

    # ── PDF → images ──
    num_pages = _pdf_to_images(pdf_path, img_dir, max_pages=preview_end if is_preview else None)
    print(f"PDF: {num_pages} pages.")

    # ── Audio ──
    print("Generating audio...")
    done_counter = [0]
    results: list[tuple[str, np.ndarray]] = []
    for slide_num in slide_keys:
        segments = script[slide_num]
        _, slide_wav = _process_slide_audio(slide_num, segments, cfg, audio_dir, total_segments, done_counter)
        results.append((slide_num, slide_wav))

    if not is_preview:
        results.sort(key=lambda x: int(x[0]))
        all_audio: list[np.ndarray] = []
        slide_silence = _silence(GAP_BETWEEN_SLIDES)
        for _, slide_wav in results:
            all_audio.extend([slide_wav, slide_silence])
        full_audio = np.concatenate(all_audio)
        audio_out = os.path.join(output_dir, f"{output_name or 'lecture'}.wav")
        sf.write(audio_out, full_audio, SAMPLE_RATE)
        print(f"Audio saved: {audio_out}")

    # ── Video clips ──
    print("Assembling clips...")
    available_images = {
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(img_dir) if f.endswith(".png")
    }
    available_audio = {
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(audio_dir) if f.endswith(".wav")
    }
    matched_slides = sorted(available_images & available_audio)

    tasks = []
    for num in matched_slides:
        key = str(num).zfill(3)
        tasks.append((
            num,
            os.path.join(img_dir, f"slide_{key}.png"),
            os.path.join(audio_dir, f"slide_{key}.wav"),
            os.path.join(video_dir, f"slide_{key}.mp4"),
            video_height,
        ))

    clip_results: list[tuple[int, str] | None] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_make_slide_clip_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            num = futures[future]
            if result:
                clip_results.append(result)
            print(f"[Video] Clip {len(clip_results)}/{len(tasks)} done (slide {num})")

    successful = sorted([r for r in clip_results if r is not None], key=lambda x: x[0])
    clip_paths = [path for _, path in successful]

    # ── Final concat ──
    video_name = f"{output_name or ('preview' if is_preview else 'lecture')}.mp4"
    video_out = os.path.join(output_dir, video_name)

    concat_list = os.path.join(work_dir, "concat_list.txt")
    with open(concat_list, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{os.path.abspath(clip)}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list, "-c", "copy", video_out,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(concat_list)

    if result.returncode != 0:
        err = result.stderr[-400:].replace("`", "'")
        print(f"❌ FFmpeg concat failed:\n{err}")
        sys.exit(1)

    size_mb = os.path.getsize(video_out) / 1024 / 1024
    print(f"✅ Job {job_id} done.")
    print(f"   Video: {video_out} ({size_mb:.1f} MB)")
    if not is_preview:
        print(f"   Audio: {audio_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ CUDA not available.")
        sys.exit(1)

    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print("Loading nanovllm VoxCPM server...")

    global server
    server = VoxCPM.from_pretrained(
        model="openbmb/VoxCPM2",
        devices=[0],
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.95,
    )
    info = server.get_model_info()
    print(f"Server ready. sr={info['sample_rate']}, encoder_sr={info['encoder_sample_rate']}")

    run_job(args.job_id, args.work_dir, args.output_dir)


if __name__ == "__main__":
    main()
