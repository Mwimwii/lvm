#!/usr/bin/env python3
"""
Lecture Video Generator — Client UI (R2 data plane)
Uploads inputs to Cloudflare R2, submits a Lightning Job, polls R2 for outputs.
Usage: python app_client.py --share
"""
import argparse, json, os, shutil, time, uuid, requests as _requests

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
CF_WORKER_URL = os.environ.get("CF_WORKER_URL", "")
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

def _r2_read_json(r2_key):
    """Read a JSON object from R2. Returns dict or None."""
    try:
        obj = _get_r2().get_object(Bucket=R2_BUCKET, Key=r2_key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None

_CF_HEADERS = {"User-Agent": "lvm-client/1.0", "Accept": "application/json"}

def _cf_base(url):
    """Strip /upload or /playlists or /health from end to get base URL."""
    for suffix in ("/upload", "/playlists", "/health"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url.rstrip("/")

def _test_cf_worker(url):
    if not url:
        return False, "No URL provided"
    try:
        r = _requests.get(f"{_cf_base(url)}/health", headers=_CF_HEADERS, timeout=10)
        return r.status_code == 200, f"HTTP {r.status_code}: {r.text[:80]}"
    except Exception as e:
        return False, f"Error: {e}"

def _fetch_playlists(url):
    if not url:
        return [], "No Worker URL set."
    try:
        r = _requests.get(f"{_cf_base(url)}/playlists", headers=_CF_HEADERS, timeout=15)
        if r.status_code != 200:
            return [], f"Worker returned {r.status_code}: {r.text[:120]}"
        items = r.json().get("playlists", [])
        return [(p["id"], p["title"]) for p in items], f"Found {len(items)} playlists."
    except Exception as e:
        return [], f"Error: {e}"

def _list_r2_jobs():
    """Scan R2 and return a list of job entries."""
    try:
        resp = _get_r2().list_objects_v2(Bucket=R2_BUCKET, Prefix="jobs/", Delimiter="/")
        prefixes = resp.get("CommonPrefixes", [])
        jobs = []
        for p in prefixes:
            batch_id = p["Prefix"].split("/")[1]
            # Look for manifest or outputs
            manifest = _r2_read_json(f"jobs/{batch_id}/manifest.json")
            if manifest and "lectures" in manifest:
                for lec in manifest["lectures"]:
                    lid = lec["id"]
                    out_keys = _r2_list(f"jobs/{batch_id}/{lid}/output/")
                    # Check status
                    status = "pending"
                    if any(k.endswith("done.json") for k in out_keys):
                        status = "done"
                    elif any(k.endswith("error.json") for k in out_keys):
                        status = "error"
                    elif out_keys:
                        status = "running"

                    # Find MP4 key
                    mp4_key = None
                    for k in out_keys:
                        if k.endswith(".mp4"):
                            mp4_key = k
                            break

                    # Find YouTube result
                    yt = _r2_read_json(f"jobs/{batch_id}/{lid}/output/yt_result.json")
                    yt_url = yt["url"] if yt and yt.get("success") else ""

                    jobs.append({
                        "batch_id": batch_id,
                        "lecture_id": lid,
                        "name": lec["name"],
                        "status": status,
                        "mp4_key": mp4_key or "",
                        "youtube_url": yt_url,
                    })
            else:
                # Legacy single-job format
                out_keys = _r2_list(f"jobs/{batch_id}/output/")
                mp4_key = next((k for k in out_keys if k.endswith(".mp4")), None)
                yt = _r2_read_json(f"jobs/{batch_id}/output/yt_result.json")
                yt_url = yt["url"] if yt and yt.get("success") else ""
                if mp4_key or out_keys:
                    jobs.append({
                        "batch_id": batch_id,
                        "lecture_id": batch_id,
                        "name": batch_id,
                        "status": "done" if mp4_key else "running",
                        "mp4_key": mp4_key or "",
                        "youtube_url": yt_url,
                    })
        return jobs
    except Exception as e:
        print(f"List R2 jobs failed: {e}")
        return []

def _get_job_log(batch_id, lecture_id):
    log_key = f"jobs/{batch_id}/{lecture_id}/output/log.txt"
    try:
        obj = _get_r2().get_object(Bucket=R2_BUCKET, Key=log_key)
        return obj["Body"].read().decode("utf-8", errors="replace")
    except Exception:
        return "No logs available."

def _download_job_explicit(batch_id, lecture_id):
    """Explicitly download a job's outputs to local disk."""
    try:
        prefix = f"jobs/{batch_id}/{lecture_id}/output/"
        out_keys = _r2_list(prefix)
        local_dir = _prepare_job_dir(f"{batch_id}_{lecture_id}")
        local_out = os.path.join(local_dir, "output")
        os.makedirs(local_out, exist_ok=True)
        downloaded = []
        for k in out_keys:
            if k.endswith(".mp4") or k.endswith(".wav"):
                lp = os.path.join(local_out, os.path.basename(k))
                _download_from_r2(k, lp)
                downloaded.append(os.path.basename(k))
        return f"Downloaded {len(downloaded)} files to {local_out}"
    except Exception as e:
        return f"Download failed: {e}"

def _prepare_job_dir(job_id):
    d = os.path.join(JOBS_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return d

# ── Batch queue helpers ───────────────────────────────────────────────────────
def _render_queue(queue):
    if not queue:
        return "Queue is empty."
    lines = ["| # | Name | Status | YouTube |", "|---|---|---|---|"]
    for i, item in enumerate(queue, 1):
        yt = item.get('youtube_url', '')
        yt_cell = f"[link]({yt})" if yt else "—"
        lines.append(f"| {i} | {item['name']} | {item['status']} | {yt_cell} |")
    return "\n".join(lines)

def _upload_and_submit_batch(queue_items, batch_job_id):
    """Upload all items in the batch to R2 and submit ONE Lightning job."""
    batch_dir = _prepare_job_dir(batch_job_id)
    
    lectures = []
    for item in queue_items:
        lid = item["job_id"] # lecture id
        ldir = os.path.join(batch_dir, lid)
        os.makedirs(ldir, exist_ok=True)
        
        shutil.copy(item["script_file"], os.path.join(ldir, "script.json"))
        shutil.copy(item["pdf_file"], os.path.join(ldir, "slides.pdf"))
        if item["ref_audio"]:
            shutil.copy(item["ref_audio"], os.path.join(ldir, "ref_voice.wav"))
        
        lectures.append({
            "id": lid,
            "name": item["name"],
            "cfg": item["cfg"],
            "timesteps": item["timesteps"],
            "video_height": item["video_height"],
            "clear_assets": item["clear_assets"],
            "preview_start": item["preview_start"],
            "preview_end": item["preview_end"],
            "ref_text": item["ref_text"],
            "youtube": item.get("youtube", {"enabled": False}),
        })

    manifest = {
        "batch_id": batch_job_id,
        "r2_bucket": R2_BUCKET,
        "lectures": lectures,
    }
    with open(os.path.join(batch_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    try:
        # Upload manifest
        _upload_to_r2(os.path.join(batch_dir, "manifest.json"), f"jobs/{batch_job_id}/manifest.json")
        
        # Upload each lecture's inputs
        for l in lectures:
            lid = l["id"]
            ldir = os.path.join(batch_dir, lid)
            _upload_to_r2(os.path.join(ldir, "script.json"), f"jobs/{batch_job_id}/{lid}/script.json")
            _upload_to_r2(os.path.join(ldir, "slides.pdf"), f"jobs/{batch_job_id}/{lid}/slides.pdf")
            if os.path.exists(os.path.join(ldir, "ref_voice.wav")):
                _upload_to_r2(os.path.join(ldir, "ref_voice.wav"), f"jobs/{batch_job_id}/{lid}/ref_voice.wav")
        
        # Upload scripts
        batch_src = os.path.join(_BASE, "app_batch_processor.py")
        bootstrap_src = os.path.join(_BASE, "app_r2_bootstrap.py")
        if os.path.exists(batch_src):
            _upload_to_r2(batch_src, f"jobs/{batch_job_id}/batch.py")
        if os.path.exists(bootstrap_src):
            _upload_to_r2(bootstrap_src, f"jobs/{batch_job_id}/bootstrap.py")
            
    except Exception as e:
        return False, f"Batch upload failed: {e}"

    env = {
        "R2_ACCESS_KEY_ID": R2_ACCESS_KEY,
        "R2_SECRET_ACCESS_KEY": R2_SECRET_KEY,
        "R2_ENDPOINT_URL": R2_ENDPOINT,
        "R2_BUCKET_NAME": R2_BUCKET,
        "JOB_ID": batch_job_id,
    }
    try:
        Job.run(
            name=f"batch-{batch_job_id}",
            machine=Machine.L40S,
            image=DOCKER_IMAGE,
            command="python /app/bootstrap.py",
            teamspace=TEAMSPACE,
            user="mpnyirongo",
            env=env,
            interruptible=True,
        )
        return True, f"Batch job {batch_job_id} submitted OK."
    except Exception as e:
        return False, f"Batch job submit failed: {e}"

def add_to_queue(queue, script_file, pdf_file, ref_audio, ref_text, cfg, timesteps,
                 video_height, clear_assets, preview_mode, preview_start,
                 preview_end, output_name, yt_title, yt_description, yt_privacy, yt_playlist_id, cf_worker_url):
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
        "youtube": {
            "enabled": True,
            "worker_url": cf_worker_url.strip() or CF_WORKER_URL,
            "title": yt_title.strip() or (output_name.strip() if output_name else f"lecture_{len(queue)+1}"),
            "description": yt_description.strip() if yt_description else "",
            "privacy": yt_privacy,
            "playlist_id": yt_playlist_id or "",
        },
    }
    queue.append(item)
    return queue, _render_queue(queue), _log(f"Added `{item['name']}` to queue (total: {len(queue)})")

def clear_queue(queue):
    queue = []
    return queue, _render_queue(queue), _log("Queue cleared.")

def submit_job(script_file, pdf_file, ref_audio, ref_text, cfg, timesteps, video_height, clear_assets,
               preview_mode, preview_start, preview_end, output_name, progress=gr.Progress()):
    _clear_log()
    
    # Just wrap this single job in a batch-of-one queue
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    item = {
        "job_id": job_id,
        "name": (output_name.strip() if output_name else f"lecture_1"),
        "script_file": script_file, "pdf_file": pdf_file,
        "ref_audio": ref_audio, "ref_text": ref_text.strip() if ref_text else "",
        "cfg": cfg, "timesteps": timesteps, "video_height": video_height,
        "clear_assets": clear_assets,
        "preview_start": int(preview_start) if preview_mode and preview_start else None,
        "preview_end": int(preview_end) if preview_mode and preview_end else None,
        "status": "pending",
        "youtube": {"enabled": False}, # single submit doesn't do auto-upload by default here
    }
    
    batch_job_id = uuid.uuid4().hex[:8]
    _log(f"Submitting one-shot batch `{batch_job_id}`...")
    
    ok, msg = _upload_and_submit_batch([item], batch_job_id)
    if not ok:
        yield None, _log(msg)
        return

    _log(msg)
    poll_interval = 30
    max_wait = 3600
    waited = 0
    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval
        
        prefix = f"jobs/{batch_job_id}/{job_id}/output/"
        out_keys = _r2_list(prefix)
        if any(k.endswith(".mp4") for k in out_keys):
            local_job_dir = _prepare_job_dir(job_id)
            local_out = os.path.join(local_job_dir, "output")
            os.makedirs(local_out, exist_ok=True)
            mp4_key = next(k for k in out_keys if k.endswith(".mp4"))
            lp = os.path.join(local_out, os.path.basename(mp4_key))
            _download_from_r2(mp4_key, lp)
            _log("Job complete!")
            yield lp, _log(f"Job complete! Video downloaded to {lp}")
            return
        
        yield None, _log(f"Polling... ({waited//60}m)")

    yield None, _log(f"Timed out after {max_wait//60}m")



def submit_all_jobs(queue, auto_upload, yt_title, yt_description, yt_privacy, cf_worker_url, progress=gr.Progress()):
    _clear_log()
    if not queue:
        yield queue, _render_queue(queue), None, _log("Queue is empty. Add jobs first.")
        return

    pending = [item for item in queue if item["status"] == "pending"]
    if not pending:
        yield queue, _render_queue(queue), None, _log("No pending jobs in queue.")
        return

    batch_job_id = uuid.uuid4().hex[:8]
    _log(f"Starting batch job `{batch_job_id}` with {len(pending)} lectures...")
    
    for item in pending:
        item["status"] = "uploading"
    yield queue, _render_queue(queue), None, _log("Uploading batch inputs...")

    ok, msg = _upload_and_submit_batch(pending, batch_job_id)
    if not ok:
        for item in pending:
            item["status"] = "failed"
        yield queue, _render_queue(queue), None, _log(msg)
        return

    for item in pending:
        item["status"] = "running"
    _log(msg)
    yield queue, _render_queue(queue), None, msg

    # Poll for batch completion item by item
    poll_interval = 30
    max_wait = 7200 # 2 hours for batch
    waited = 0
    
    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval
        
        # Check status of each item by looking for its output done.json or MP4
        any_pending = False
        for item in pending:
            if item["status"] == "done":
                continue
            
            # Look for output files in R2 under batch_job_id/lecture_id/output/
            prefix = f"jobs/{batch_job_id}/{item['job_id']}/output/"
            out_keys = _r2_list(prefix)
            
            if any(k.endswith(".mp4") for k in out_keys):
                mp4_key = next(k for k in out_keys if k.endswith(".mp4"))
                item["public_url"] = _r2_presigned_url(mp4_key)
                item["status"] = "done"
                _log(f"Lecture `{item['name']}` finished.")
                
                # Check for YouTube result if enabled
                if auto_upload:
                    res_key = f"{prefix}yt_result.json"
                    yt_res = _r2_read_json(res_key)
                    if yt_res:
                        if yt_res.get("success"):
                            item["youtube_url"] = yt_res["url"]
                            _log(f"  YouTube: {yt_res['url']}")
                        else:
                            _log(f"  YouTube failed: {yt_res.get('error')}")
            else:
                any_pending = True
        
        yield queue, _render_queue(queue), None, f"Batch running... ({waited//60}m)"
        
        if not any_pending:
            break

    summary = f"Batch `{batch_job_id}` complete."
    _log(summary)
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
    mp4_key = next(k for k in out_keys if k.endswith(".mp4"))
    url = _r2_presigned_url(mp4_key)
    _log(f"Video ready: {url}")
    # Only download if user explicitly requests it later
    yield None, _log(f"Job {job_id} recovered! Video URL: {url}")


def build_ui():
    with gr.Blocks(title="Lecture Video Generator - Client") as demo:
        gr.Markdown("# Lecture Video Generator — Client (R2)")

        # ── Shared state ─────────────────────────────────────────────────────
        queue_state = gr.State(value=[])
        playlists_state = gr.State(value=[])  # [(id, title), ...]

        with gr.Sidebar():
            gr.Markdown("## Navigation")
            btn_submit   = gr.Button("Submit",   variant="primary")
            btn_explorer = gr.Button("Explorer", variant="secondary")
            btn_settings = gr.Button("Settings", variant="secondary")

        # ── SUBMIT VIEW ───────────────────────────────────────────────────────
        with gr.Column(visible=True) as col_submit:
            gr.Markdown("## Lecture Inputs")
            with gr.Row():
                script_input = gr.File(label="script.json", file_types=[".json"])
                pdf_input    = gr.File(label="slides.pdf",  file_types=[".pdf"])

            gr.Markdown("## Generation Settings")
            with gr.Row():
                cfg_slider       = gr.Slider(minimum=1.0, maximum=5.0, value=2.7, step=0.1, label="CFG")
                timesteps_slider = gr.Slider(minimum=5, maximum=25, value=10, step=1, label="TTS Steps")
                video_height     = gr.Dropdown(choices=[480, 720, 1080], value=720, label="Height")
            with gr.Row():
                clear_checkbox = gr.Checkbox(value=True,  label="Clear previous assets")
                preview_mode   = gr.Checkbox(value=False, label="Preview mode")
            with gr.Row():
                preview_start = gr.Number(value=1, minimum=1, step=1, label="Preview start")
                preview_end   = gr.Number(value=3, minimum=1, step=1, label="Preview end")

            output_name = gr.Textbox(label="Output filename (no ext)", placeholder="e.g., lecture_week1")

            # ── YouTube ───────────────────────────────────────────────────────
            with gr.Accordion("YouTube Upload", open=False):
                gr.Markdown("*Fetch playlists in the Settings tab first.*")
                with gr.Row():
                    yt_title   = gr.Textbox(label="Title", placeholder="e.g., Week 1 — Introduction")
                    yt_privacy = gr.Dropdown(choices=["private","unlisted","public"], value="private", label="Privacy")
                yt_description = gr.Textbox(label="Description", lines=2)
                yt_playlist    = gr.Dropdown(label="Playlist", choices=[], interactive=True)
                auto_upload    = gr.Checkbox(value=False, label="Auto-upload after render")

            gr.Markdown("---")
            submit_btn    = gr.Button("Submit L40S Job",        variant="primary")
            add_queue_btn = gr.Button("Add to Batch Queue",     variant="secondary")

            gr.Markdown("## Batch Queue")
            queue_md      = gr.Markdown("Queue is empty.")
            with gr.Row():
                clear_queue_btn = gr.Button("Clear Queue",    variant="secondary")
                submit_all_btn  = gr.Button("Submit All Jobs",variant="primary")

        # ── EXPLORER VIEW ─────────────────────────────────────────────────────
        with gr.Column(visible=False) as col_explorer:
            gr.Markdown("## R2 Job Explorer")
            explorer_refresh = gr.Button("Refresh", variant="secondary")
            explorer_df = gr.Dataframe(
                headers=["Batch", "Lecture", "Name", "Status", "Video", "YouTube"],
                label="Jobs", interactive=False,
            )
            with gr.Row():
                explorer_selected_batch   = gr.Textbox(label="Batch ID",   interactive=False)
                explorer_selected_lecture = gr.Textbox(label="Lecture ID", interactive=False)
            explorer_log = gr.Textbox(label="Logs", lines=12, interactive=False)
            with gr.Row():
                explorer_download_btn    = gr.Button("Download Selected", variant="secondary")
                explorer_download_status = gr.Markdown()

            def refresh_explorer():
                rows = []
                for j in _list_r2_jobs():
                    v = f"[Open]({_r2_presigned_url(j['mp4_key'])})" if j["mp4_key"] else "—"
                    y = f"[Open]({j['youtube_url']})"              if j["youtube_url"] else "—"
                    rows.append([j["batch_id"], j["lecture_id"], j["name"], j["status"], v, y])
                return rows

            def on_select_row(evt):
                if evt is None or not hasattr(evt, "index"):
                    return "", "", "No row selected."
                jobs = _list_r2_jobs()
                idx  = evt.index[0]
                if idx < len(jobs):
                    j = jobs[idx]
                    return j["batch_id"], j["lecture_id"], _get_job_log(j["batch_id"], j["lecture_id"])
                return "", "", ""

            def do_download(batch_id, lecture_id):
                if not batch_id or not lecture_id:
                    return "Select a row first."
                return _download_job_explicit(batch_id, lecture_id)

            explorer_refresh.click(fn=refresh_explorer, outputs=explorer_df)
            explorer_df.select(fn=on_select_row, outputs=[explorer_selected_batch, explorer_selected_lecture, explorer_log])
            explorer_download_btn.click(fn=do_download, inputs=[explorer_selected_batch, explorer_selected_lecture], outputs=explorer_download_status)

        # ── SETTINGS VIEW ─────────────────────────────────────────────────────
        with gr.Column(visible=False) as col_settings:
            gr.Markdown("## Reference Voice")
            ref_status = gr.Markdown("No reference voice saved.")
            with gr.Row():
                ref_audio = gr.Audio(label="Reference Audio (.wav)", type="filepath", sources=["upload"])
                ref_text  = gr.Textbox(label="Reference Transcript", lines=3)
            save_ref_btn = gr.Button("Save Reference Voice", variant="secondary")
            save_ref_btn.click(fn=save_reference, inputs=[ref_audio, ref_text], outputs=ref_status)

            gr.Markdown("---")
            gr.Markdown("## Cloudflare Worker")
            with gr.Row():
                cf_worker_url  = gr.Textbox(label="Worker URL", value=CF_WORKER_URL, scale=3)
                test_worker_btn = gr.Button("Test", variant="secondary", scale=1)
            worker_status = gr.Markdown()
            test_worker_btn.click(
                fn=lambda url: _test_cf_worker(url)[1],
                inputs=cf_worker_url, outputs=worker_status,
            )

            gr.Markdown("## YouTube Playlists")
            fetch_playlists_btn = gr.Button("Fetch My Playlists", variant="secondary")
            playlist_status_md  = gr.Markdown()
            fetch_playlists_btn.click(
                fn=lambda url: _do_fetch(url),
                inputs=cf_worker_url,
                outputs=[yt_playlist, playlists_state, playlist_status_md],
            )

            gr.Markdown("---")
            gr.Markdown("## Resume Previous Job")
            with gr.Row():
                job_id_input = gr.Textbox(label="Batch ID", placeholder="e.g., a1b2c3d4")
                check_btn    = gr.Button("Poll Job", variant="secondary")

        # ── STATUS AREA ───────────────────────────────────────────────────────
        gr.Markdown("---")
        status_md   = gr.Markdown("*Status will appear here…*")
        output_video = gr.Video(label="Output video", visible=False)

        # ── HELPERS that need UI refs ─────────────────────────────────────────
        def _do_fetch(url):
            items, msg = _fetch_playlists(url)
            choices = [(f"{title}  [{pid}]", pid) for pid, title in items]
            return gr.update(choices=choices, value=None), items, msg

        # ── NAVIGATION ───────────────────────────────────────────────────────
        _nav_outs = [col_submit, col_explorer, col_settings, btn_submit, btn_explorer, btn_settings]

        btn_submit.click(
            fn=lambda: (gr.update(visible=True),  gr.update(visible=False), gr.update(visible=False),
                        gr.update(variant="primary"), gr.update(variant="secondary"), gr.update(variant="secondary")),
            outputs=_nav_outs,
        )
        btn_explorer.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True),  gr.update(visible=False),
                        gr.update(variant="secondary"), gr.update(variant="primary"), gr.update(variant="secondary")),
            outputs=_nav_outs,
        )
        btn_settings.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                        gr.update(variant="secondary"), gr.update(variant="secondary"), gr.update(variant="primary")),
            outputs=_nav_outs,
        )

        # ── EVENT BINDINGS ───────────────────────────────────────────────────
        submit_btn.click(
            fn=submit_job,
            inputs=[script_input, pdf_input, ref_audio, ref_text,
                    cfg_slider, timesteps_slider, video_height,
                    clear_checkbox, preview_mode, preview_start, preview_end, output_name],
            outputs=[output_video, status_md],
            show_progress="full",
        )
        add_queue_btn.click(
            fn=add_to_queue,
            inputs=[queue_state, script_input, pdf_input, ref_audio, ref_text,
                    cfg_slider, timesteps_slider, video_height,
                    clear_checkbox, preview_mode, preview_start, preview_end, output_name,
                    yt_title, yt_description, yt_privacy, yt_playlist, cf_worker_url],
            outputs=[queue_state, queue_md, status_md],
        )
        clear_queue_btn.click(fn=clear_queue, inputs=queue_state, outputs=[queue_state, queue_md, status_md])
        submit_all_btn.click(
            fn=submit_all_jobs,
            inputs=[queue_state, auto_upload, yt_title, yt_description, yt_privacy, cf_worker_url],
            outputs=[queue_state, queue_md, output_video, status_md],
            show_progress="full",
        )
        check_btn.click(fn=check_or_resume_job, inputs=job_id_input, outputs=[output_video, status_md], show_progress="full")

    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share)

if __name__ == "__main__":
    main()
