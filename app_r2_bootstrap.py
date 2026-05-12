#!/usr/bin/env python3
"""
R2 bootstrap — runs inside the Lightning Job container.
Downloads inputs from R2, runs app_batch_processor.py, uploads outputs back.
Env vars required: R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, JOB_ID
"""

import os
import sys
import glob
import shutil
import subprocess
import json
import urllib.request

import boto3
from botocore.config import Config as BotoConfig


R2 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
    config=BotoConfig(signature_version="s3v4"),
)

BID = os.environ["JOB_ID"]
BUCKET = os.environ["R2_BUCKET_NAME"]


def r2_download(r2_key: str, local_path: str) -> None:
    print(f"[bootstrap] download {r2_key} -> {local_path}")
    R2.download_file(BUCKET, r2_key, local_path)


def r2_upload(local_path: str, r2_key: str) -> None:
    print(f"[bootstrap] upload {local_path} -> {r2_key}")
    R2.upload_file(local_path, BUCKET, r2_key)


def notify_youtube_worker(lid, video_name, youtube_meta):
    cf_url = youtube_meta.get("worker_url")
    if not cf_url:
        print(f"[bootstrap] [{lid}] YouTube enabled but no worker_url")
        return

    payload = json.dumps({
        "r2_key": f"jobs/{BID}/{lid}/output/{video_name}",
        "title": youtube_meta.get("title", "Lecture Video"),
        "description": youtube_meta.get("description", ""),
        "privacy": youtube_meta.get("privacy", "private"),
        "playlist_id": youtube_meta.get("playlist_id", ""),
        "callback_key": f"jobs/{BID}/{lid}/output/yt_result.json",
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            cf_url, data=payload, method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "lvm-bootstrap/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"[bootstrap] [{lid}] CF Worker notified: {resp.read().decode('utf-8')[:100]}")
    except Exception as e:
        print(f"[bootstrap] [{lid}] CF Worker failed: {e}")

def upload_log(lid, log_text):
    try:
        R2.put_object(Bucket=BUCKET, Key=f"jobs/{BID}/{lid}/output/log.txt", Body=log_text.encode())
    except Exception as e:
        print(f"[bootstrap] [{lid}] log upload failed: {e}")


def main():
    print(f"[bootstrap] Batch Job {BID} starting...")

    # 1. Download Batch Manifest
    manifest_path = "/tmp/manifest.json"
    r2_download(f"jobs/{BID}/manifest.json", manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # 2. Get batch scripts
    if os.path.exists("/app/batch.py"):
        shutil.copy("/app/batch.py", "/tmp/batch.py")
    else:
        r2_download(f"jobs/{BID}/batch.py", "/tmp/batch.py")

    lectures = manifest.get("lectures", [])
    print(f"[bootstrap] Found {len(lectures)} lectures in batch.")

    for i, l in enumerate(lectures, 1):
        lid = l["id"]
        print(f"\n[bootstrap] [{i}/{len(lectures)}] Processing lecture {lid} ({l['name']})...")
        
        # Cleanup tmp dirs for this lecture
        work_dir = f"/tmp/{lid}"
        out_dir = f"{work_dir}/output"
        if os.path.exists(work_dir): shutil.rmtree(work_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Download lecture inputs
        r2_download(f"jobs/{BID}/{lid}/script.json", f"{work_dir}/script.json")
        r2_download(f"jobs/{BID}/{lid}/slides.pdf", f"{work_dir}/slides.pdf")
        
        # Write specific manifest for the processor
        l_manifest = {
            "job_id": lid,
            "cfg": l["cfg"],
            "timesteps": l["timesteps"],
            "video_height": l["video_height"],
            "clear_assets": l["clear_assets"],
            "preview_start": l["preview_start"],
            "preview_end": l["preview_end"],
            "ref_text": l["ref_text"],
            "output_name": l["name"],
        }
        with open(f"{work_dir}/manifest.json", "w") as f:
            json.dump(l_manifest, f)

        # Optional ref voice
        try:
            r2_download(f"jobs/{BID}/{lid}/ref_voice.wav", f"{work_dir}/ref_voice.wav")
        except: pass

        # Run processor and capture logs
        log_lines = []
        try:
            proc = subprocess.run(
                [sys.executable, "/tmp/batch.py", "--job-id", lid, "--work-dir", work_dir, "--output-dir", out_dir],
                capture_output=True, text=True, check=True,
            )
            log_lines.append(proc.stdout)
            if proc.stderr:
                log_lines.append(proc.stderr)
            
            # Upload outputs for THIS lecture
            print(f"[bootstrap] [{lid}] Uploading outputs...")
            mp4_name = ""
            for fp in glob.glob(f"{out_dir}/*"):
                fname = os.path.basename(fp)
                if fname.endswith(".mp4"): mp4_name = fname
                r2_upload(fp, f"jobs/{BID}/{lid}/output/{fname}")

            # Notify YouTube Worker if enabled
            yt = l.get("youtube")
            if yt and yt.get("enabled") and mp4_name:
                notify_youtube_worker(lid, mp4_name, yt)

            # Done marker + logs for this lecture
            log_text = "\n".join(log_lines)
            R2.put_object(Bucket=BUCKET, Key=f"jobs/{BID}/{lid}/output/done.json", Body=b'{"done":true}')
            upload_log(lid, log_text)
            
        except Exception as e:
            print(f"[bootstrap] [{lid}] FAILED: {e}")
            log_lines.append(str(e))
            R2.put_object(Bucket=BUCKET, Key=f"jobs/{BID}/{lid}/output/error.json", Body=json.dumps({"error": str(e)}).encode())
            upload_log(lid, "\n".join(log_lines))

    print("\n[bootstrap] Batch complete.")


if __name__ == "__main__":
    main()
