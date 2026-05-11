#!/usr/bin/env python3
"""
Lecture Video Generator — Client UI (R2 data plane)
Uploads inputs to Cloudflare R2, submits a Lightning Job, polls R2 for outputs.
Usage: python app_client.py --share
"""
import argparse, json, os, shutil, time, uuid

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ImportError:
    pass

import gradio as gr
from lightning_sdk import Job, Machine
import boto3
from botocore.config import Config as BotoConfig

_BASE = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(_BASE, "client_jobs")
DOCKER_IMAGE = "mpnyirongo/lvm-processor:latest"
TEAMSPACE = "language-model"
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT   = os.environ.get("R2_ENDPOINT_URL", "")
R2_BUCKET     = os.environ.get("R2_BUCKET_NAME", "")
_LOG_LINES = []
_r2 = None

def _get_r2():
    global _r2
    if _r2 is None:
        _r2 = boto3.client("s3", endpoint_url=R2_ENDPOINT, aws_access_key_id=R2_ACCESS_KEY,
                           aws_secret_access_key=R2_SECRET_KEY, region_name="auto",
                           config=BotoConfig(signature_version="s3v4"))
    return _r2

def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    _LOG_LINES.append(line)
    print(line)
    return "\n".join(_LOG_LINES[-40:])

def _clear_log():
    _LOG_LINES.clear()

def _ensure_dirs():
    os.makedirs(JOBS_DIR, exist_ok=True)

def save_reference(audio_file, text):
    _ensure_dirs()
    if audio_file is None:
        return "Please upload a reference audio file."
    dest = os.path.join(JOBS_DIR, "ref_voice.wav")
    shutil.copy(audio_file, dest)
    return f"Reference saved: {dest} ({os.path.getsize(dest)//1024//1024:.1f} MB)"

def _r2_key(job_id, path):
    return f"jobs/{job_id}/{path}"

def _upload_to_r2(local_path, r2_key):
    _get_r2().upload_file(local_path, R2_BUCKET, r2_key)

def _r2_exists(r2_key):
    try:
        _get_r2().head_object(Bucket=R2_BUCKET, Key=r2_key)
        return True
    except Exception:
        return False

def _r2_list(prefix):
    try:
        resp = _get_r2().list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
        return [o["Key"] for o in resp.get("Contents", [])]
    except Exception:
        return []

def _download_from_r2(r2_key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _get_r2().download_file(R2_BUCKET, r2_key, local_path)

def _r2_presigned_url(r2_key, expires=604800):
    """Return a public presigned URL for an R2 object (default 7 days)."""
    try:
        url = _get_r2().generate_presigned_url(
            "get_object", Params={"Bucket": R2_BUCKET, "Key": r2_key}, ExpiresIn=expires
        )
        return url
    except Exception as e:
        return f"(presign failed: {e})"

def _prepare_job_dir(job_id):
    d = os.path.join(JOBS_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return d

# ── Batch queue helpers ───────────────────────────────────────────────────────
def _render_queue(queue):
    if not queue:
        return "Queue is empty."
    lines = ["| # | Name | Job ID | Status |", "|---|---|---|---|"]
    for i, item in enumerate(queue, 1):
        lines.append(f"| {i} | {item['name']} | `{item['job_id']}` | {item['status']} |")
    return "\n".join(lines)

def _upload_and_submit_item(item):
    """Upload one queued item to R2 and submit Lightning job. Returns (ok, msg)."""
    job_id = item["job_id"]
    job_dir = _prepare_job_dir(job_id)

    shutil.copy(item["script_file"], os.path.join(job_dir, "script.json"))
    shutil.copy(item["pdf_file"], os.path.join(job_dir, "slides.pdf"))
    if item["ref_audio"] is not None:
        shutil.copy(item["ref_audio"], os.path.join(job_dir, "ref_voice.wav"))

    manifest = {
        "job_id": job_id,
        "cfg": item["cfg"],
        "timesteps": item["timesteps"],
        "video_height": item["video_height"],
        "clear_assets": item["clear_assets"],
        "preview_start": item["preview_start"],
        "preview_end": item["preview_end"],
        "ref_text": item["ref_text"],
        "output_name": item["name"],
        "r2_bucket": R2_BUCKET,
    }
    with open(os.path.join(job_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    try:
        _upload_to_r2(os.path.join(job_dir, "script.json"), _r2_key(job_id, "inputs/script.json"))
        _upload_to_r2(os.path.join(job_dir, "slides.pdf"), _r2_key(job_id, "inputs/slides.pdf"))
        _upload_to_r2(os.path.join(job_dir, "manifest.json"), _r2_key(job_id, "inputs/manifest.json"))
        batch_src = os.path.join(_BASE, "app_batch_processor.py")
        bootstrap_src = os.path.join(_BASE, "app_r2_bootstrap.py")
        if os.path.exists(batch_src):
            _upload_to_r2(batch_src, _r2_key(job_id, "batch.py"))
        if os.path.exists(bootstrap_src):
            _upload_to_r2(bootstrap_src, _r2_key(job_id, "bootstrap.py"))
        if os.path.exists(os.path.join(job_dir, "ref_voice.wav")):
            _upload_to_r2(os.path.join(job_dir, "ref_voice.wav"), _r2_key(job_id, "inputs/ref_voice.wav"))
    except Exception as e:
        return False, f"Job {job_id} upload failed: {e}"

    env = {
        "R2_ACCESS_KEY_ID": R2_ACCESS_KEY,
        "R2_SECRET_ACCESS_KEY": R2_SECRET_KEY,
        "R2_ENDPOINT_URL": R2_ENDPOINT,
        "R2_BUCKET_NAME": R2_BUCKET,
        "JOB_ID": job_id,
    }
    try:
        Job.run(
            name=job_id,
            machine=Machine.L40S,
            image=DOCKER_IMAGE,
            command="python /app/bootstrap.py",
            teamspace=TEAMSPACE,
            user="mpnyirongo",
            env=env,
            interruptible=True,
        )
        return True, f"Job {job_id} submitted OK."
    except Exception as e:
        return False, f"Job {job_id} submit failed: {e}"

def add_to_queue(queue, script_file, pdf_file, ref_audio, ref_text, cfg, timesteps,
                 video_height, clear_assets, preview_mode, preview_start,
                 preview_end, output_name):
    queue = queue or []
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    item = {
        "job_id": job_id,
        "name": (output_name.strip() if output_name else f"lecture_{len(queue)+1}"),
        "script_file": script_file, "pdf_file": pdf_file,
        "ref_audio": ref_audio, "ref_text": ref_text.strip() if ref_text else "",
        "cfg": cfg, "timesteps": timesteps, "video_height": video_height,
        "clear_assets": clear_assets,
        "preview_start": int(preview_start) if preview_mode and preview_start else None,
        "preview_end": int(preview_end) if preview_mode and preview_end else None,
        "status": "pending",
    }
    queue.append(item)
    return queue, _render_queue(queue), _log(f"Added `{item['name']}` to queue (total: {len(queue)})")

def clear_queue(queue):
    queue = []
    return queue, _render_queue(queue), _log("Queue cleared.")

def submit_job(script_file, pdf_file, ref_audio, ref_text, cfg, timesteps, video_height, clear_assets,
               preview_mode, preview_start, preview_end, output_name, progress=gr.Progress()):
    _clear_log()

    if not all([R2_ACCESS_KEY, R2_SECRET_KEY, R2_ENDPOINT, R2_BUCKET]):
        yield None, _log("R2 credentials missing. Check .env file.")
        return
    if script_file is None:
        yield None, _log("Please upload script.json.")
        return
    if pdf_file is None:
        yield None, _log("Please upload slides.pdf.")
        return

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_dir = _prepare_job_dir(job_id)
    _log(f"Job ID: {job_id}")
    _log(f"Local dir: {job_dir}")

    progress(0.05, desc="Uploading inputs to R2...")
    shutil.copy(script_file, os.path.join(job_dir, "script.json"))
    shutil.copy(pdf_file, os.path.join(job_dir, "slides.pdf"))
    if ref_audio is not None:
        shutil.copy(ref_audio, os.path.join(job_dir, "ref_voice.wav"))
        _log("Copied ref_voice.wav from UI.")

    manifest = {
        "job_id": job_id, "cfg": cfg, "timesteps": timesteps,
        "video_height": video_height, "clear_assets": clear_assets,
        "preview_start": int(preview_start) if preview_mode and preview_start else None,
        "preview_end": int(preview_end) if preview_mode and preview_end else None,
        "ref_text": ref_text.strip() if ref_text else "",
        "output_name": output_name.strip() if output_name else None,
        "r2_bucket": R2_BUCKET,
    }
    with open(os.path.join(job_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    _log("Manifest written.")

    batch_src = os.path.join(_BASE, "app_batch_processor.py")
    bootstrap_src = os.path.join(_BASE, "app_r2_bootstrap.py")
    if not os.path.exists(batch_src):
        yield None, _log("ERROR: app_batch_processor.py not found")
        return
    if not os.path.exists(bootstrap_src):
        yield None, _log("ERROR: app_r2_bootstrap.py not found")
        return
    shutil.copy(batch_src, os.path.join(job_dir, "app_batch_processor.py"))
    shutil.copy(bootstrap_src, os.path.join(job_dir, "app_r2_bootstrap.py"))

    progress(0.1, desc="Uploading to R2...")
    try:
        _upload_to_r2(os.path.join(job_dir, "script.json"), _r2_key(job_id, "inputs/script.json"))
        _upload_to_r2(os.path.join(job_dir, "slides.pdf"), _r2_key(job_id, "inputs/slides.pdf"))
        _upload_to_r2(os.path.join(job_dir, "manifest.json"), _r2_key(job_id, "inputs/manifest.json"))
        _upload_to_r2(os.path.join(job_dir, "app_batch_processor.py"), _r2_key(job_id, "batch.py"))
        _upload_to_r2(os.path.join(job_dir, "app_r2_bootstrap.py"), _r2_key(job_id, "bootstrap.py"))
        if os.path.exists(os.path.join(job_dir, "ref_voice.wav")):
            _upload_to_r2(os.path.join(job_dir, "ref_voice.wav"), _r2_key(job_id, "inputs/ref_voice.wav"))
        _log("All inputs uploaded to R2.")
    except Exception as e:
        yield None, _log(f"R2 upload failed: {e}")
        return

    progress(0.2, desc="Submitting L40S Docker Job...")
    _log("Launching Docker-based Job on L40S...")

    env = {
        "R2_ACCESS_KEY_ID": R2_ACCESS_KEY,
        "R2_SECRET_ACCESS_KEY": R2_SECRET_KEY,
        "R2_ENDPOINT_URL": R2_ENDPOINT,
        "R2_BUCKET_NAME": R2_BUCKET,
        "JOB_ID": job_id,
    }

    try:
        job = Job.run(
            name=job_id,
            machine=Machine.L40S,
            image=DOCKER_IMAGE,
            command="python /app/bootstrap.py",
            teamspace=TEAMSPACE,
            user="mpnyirongo",
            env=env,
            interruptible=True,
        )
        _log(f"Job.run() returned. Status: {job.status}")
    except Exception as e:
        yield None, _log(f"Job.run() FAILED: {e}")
        return

    progress(0.3, desc="Job launched. Polling R2 every 30s...")
    _log("Beginning poll loop.")

    poll_interval = 30
    max_wait = 3600
    waited = 0
    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval
        _log(f"Poll #{waited//poll_interval} - {waited}s elapsed")
        out_keys = _r2_list(f"jobs/{job_id}/output/")
        _log(f"  R2 output keys: {out_keys}")
        if any(k.endswith(".mp4") for k in out_keys):
            progress(0.9, desc="Outputs found! Downloading...")
            _log("Outputs detected. Downloading...")
            local_job_dir = _prepare_job_dir(job_id)
            local_out = os.path.join(local_job_dir, "output")
            os.makedirs(local_out, exist_ok=True)
            for k in out_keys:
                fname = os.path.basename(k)
                lp = os.path.join(local_out, fname)
                _download_from_r2(k, lp)
                _log(f"  Downloaded: {fname}")
            video_path = os.path.join(local_out, next((os.path.basename(k) for k in out_keys if k.endswith(".mp4")), ""))
            progress(1.0, desc="Finished!")
            yield (video_path if os.path.exists(video_path) else None), (
                _log(f"Job {job_id} complete!") + f"\n\nVideo: `{video_path}`\nL40S terminated."
            )
            return
        else:
            minutes = waited // 60
            progress(0.3 + (0.6 * min(waited, 600) / 600), desc=f"Running... ({minutes}m)")

    yield None, _log(f"Timed out after {max_wait//60}m")


def submit_all_jobs(queue, progress=gr.Progress()):
    _clear_log()
    if not queue:
        yield queue, _render_queue(queue), None, _log("Queue is empty. Add jobs first.")
        return

    total = len(queue)
    pending = [item for item in queue if item["status"] == "pending"]
    if not pending:
        yield queue, _render_queue(queue), None, _log("All jobs already processed.")
        return

    _log(f"Sequential batch: {len(pending)} jobs to run one-at-a-time (saves cost)")
    poll_interval = 30
    max_wait = 3600  # 1 hour per job

    for idx, item in enumerate(pending, 1):
        _log(f"\n--- [{idx}/{len(pending)}] Starting {item['name']} ---")
        item["status"] = "uploading"
        ok, msg = _upload_and_submit_item(item)
        _log(msg)
        if not ok:
            item["status"] = "failed"
            yield queue, _render_queue(queue), None, msg
            continue

        item["status"] = "running"
        yield queue, _render_queue(queue), None, msg

        # Poll this single job until done
        waited = 0
        done = False
        while waited < max_wait and not done:
            time.sleep(poll_interval)
            waited += poll_interval
            out_keys = _r2_list(f"jobs/{item['job_id']}/output/")
            if any(k.endswith(".mp4") for k in out_keys):
                # Download outputs
                local_job_dir = _prepare_job_dir(item["job_id"])
                local_out = os.path.join(local_job_dir, "output")
                os.makedirs(local_out, exist_ok=True)
                for k in out_keys:
                    _download_from_r2(k, os.path.join(local_out, os.path.basename(k)))
                # Generate presigned URL for the MP4
                mp4_key = next((k for k in out_keys if k.endswith(".mp4")), None)
                if mp4_key:
                    url = _r2_presigned_url(mp4_key)
                    item["public_url"] = url
                    _log(f"Public link: {url}")
                item["status"] = "done"
                _log(f"Done: {item['name']}")
                done = True
            else:
                minutes = waited // 60
                progress(idx / total, desc=f"Job {idx}/{total} running ({minutes}m)")
                status_msg = f"Poll {waited//30}: waiting for {item['name']}"
                _log(status_msg)
                yield queue, _render_queue(queue), None, status_msg

        if not done:
            item["status"] = "timed out"
            _log(f"Timed out: {item['name']}")

    done_count = sum(1 for i in queue if i["status"] == "done")
    failed_count = sum(1 for i in queue if i["status"] == "failed")
    timed_out_count = sum(1 for i in queue if i["status"] == "timed out")
    summary = f"Batch complete: {done_count} done, {failed_count} failed, {timed_out_count} timed out"
    _log(summary)
    # Collect all public URLs
    urls = [f"- **{i['name']}**: [{i.get('public_url', 'n/a')[:60]}...]({i.get('public_url', '#')})" for i in queue if 'public_url' in i]
    if urls:
        _log("\n### Public Links\n" + "\n".join(urls))
    yield queue, _render_queue(queue), None, summary


def check_or_resume_job(job_id_input, progress=gr.Progress()):
    job_id = job_id_input.strip()
    if not job_id:
        yield None, _log("Please enter a Job ID.")
        return
    _clear_log()
    _log(f"Checking job {job_id} on R2...")
    out_keys = _r2_list(f"jobs/{job_id}/output/")
    _log(f"Found keys: {out_keys}")
    if not any(k.endswith(".mp4") for k in out_keys):
        yield None, _log("No outputs yet. Job may still be running.")
        return
    local_job_dir = _prepare_job_dir(job_id)
    local_out = os.path.join(local_job_dir, "output")
    os.makedirs(local_out, exist_ok=True)
    for k in out_keys:
        fname = os.path.basename(k)
        lp = os.path.join(local_out, fname)
        _download_from_r2(k, lp)
        _log(f"Downloaded: {fname}")
    video_path = os.path.join(local_out, next((os.path.basename(k) for k in out_keys if k.endswith(".mp4")), ""))
    yield (video_path if os.path.exists(video_path) else None), _log(f"Job {job_id} recovered!")


def build_ui():
    with gr.Blocks(title="Lecture Video Generator - Client") as demo:
        gr.Markdown("# Lecture Video Generator - Client (R2)")
        gr.Markdown("Submits one-shot Docker Jobs to L40S via Cloudflare R2 data plane.")

        with gr.Accordion("Reference Voice Setup", open=True):
            ref_status = gr.Markdown(value="No custom reference voice set.")
            with gr.Row():
                ref_audio = gr.Audio(label="Upload Reference Audio (.wav)", type="filepath", sources=["upload"])
                ref_text = gr.Textbox(label="Reference Transcript", lines=3)
            save_ref_btn = gr.Button("Save Reference Voice", variant="secondary")
            save_ref_btn.click(fn=save_reference, inputs=[ref_audio, ref_text], outputs=ref_status)

        gr.Markdown("## Lecture Inputs")
        with gr.Row():
            script_input = gr.File(label="script.json", file_types=[".json"])
            pdf_input = gr.File(label="slides.pdf", file_types=[".pdf"])

        gr.Markdown("## Generation Settings")
        with gr.Row():
            cfg_slider = gr.Slider(minimum=1.0, maximum=5.0, value=2.7, step=0.1, label="CFG")
            timesteps_slider = gr.Slider(minimum=5, maximum=25, value=10, step=1, label="TTS Steps")
            video_height = gr.Dropdown(choices=[480, 720, 1080], value=720, label="Height")
        with gr.Row():
            clear_checkbox = gr.Checkbox(value=True, label="Clear previous assets")
            preview_mode = gr.Checkbox(value=False, label="Preview mode (subset only)")
        with gr.Row():
            preview_start = gr.Number(value=1, minimum=1, step=1, label="Preview start")
            preview_end = gr.Number(value=3, minimum=1, step=1, label="Preview end")

        gr.Markdown("## Output Naming")
        output_name = gr.Textbox(label="Filename (no ext)", placeholder="e.g., lecture_week1")

        submit_btn = gr.Button("Submit L40S Job", variant="primary")

        # ── Batch Queue ──────────────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("## Batch Queue")
        queue_state = gr.State(value=[])
        queue_md = gr.Markdown("Queue is empty.")
        with gr.Row():
            add_queue_btn = gr.Button("Add Current Job to Queue", variant="secondary")
            clear_queue_btn = gr.Button("Clear Queue", variant="secondary")
        submit_all_btn = gr.Button("Submit All Jobs", variant="primary")

        gr.Markdown("## Resume / Check Previous Job")
        with gr.Row():
            job_id_input = gr.Textbox(label="Job ID", placeholder="e.g., job_a1b2c3d4")
            check_btn = gr.Button("Check / Resume", variant="secondary")

        gr.Markdown("## Job Result")
        output_video = gr.Video(label="Output video")
        status_md = gr.Markdown("*Status will appear here...*")

        submit_btn.click(
            fn=submit_job,
            inputs=[script_input, pdf_input, ref_audio, ref_text, cfg_slider, timesteps_slider, video_height,
                    clear_checkbox, preview_mode, preview_start, preview_end, output_name],
            outputs=[output_video, status_md],
            show_progress="full",
        )
        add_queue_btn.click(
            fn=add_to_queue,
            inputs=[queue_state, script_input, pdf_input, ref_audio, ref_text, cfg_slider, timesteps_slider, video_height,
                    clear_checkbox, preview_mode, preview_start, preview_end, output_name],
            outputs=[queue_state, queue_md, status_md],
        )
        clear_queue_btn.click(
            fn=clear_queue,
            inputs=[queue_state],
            outputs=[queue_state, queue_md, status_md],
        )
        submit_all_btn.click(
            fn=submit_all_jobs,
            inputs=[queue_state],
            outputs=[queue_state, queue_md, output_video, status_md],
            show_progress="full",
        )
        check_btn.click(
            fn=check_or_resume_job,
            inputs=[job_id_input],
            outputs=[output_video, status_md],
            show_progress="full",
        )
    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share)

if __name__ == "__main__":
    main()
