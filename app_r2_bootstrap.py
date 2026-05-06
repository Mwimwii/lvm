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


def main():
    print(f"[bootstrap] Job {BID} starting...")

    os.makedirs("/tmp/output", exist_ok=True)

    # Use embedded batch processor if available (Docker mode), else download from R2
    if os.path.exists("/app/batch.py"):
        print("[bootstrap] Using embedded /app/batch.py (Docker mode)")
        shutil.copy("/app/batch.py", "/tmp/batch.py")
    else:
        r2_download(f"jobs/{BID}/batch.py", "/tmp/batch.py")

    r2_download(f"jobs/{BID}/inputs/script.json", "/tmp/script.json")
    r2_download(f"jobs/{BID}/inputs/slides.pdf", "/tmp/slides.pdf")
    r2_download(f"jobs/{BID}/inputs/manifest.json", "/tmp/manifest.json")

    # Optional ref voice
    try:
        r2_download(f"jobs/{BID}/inputs/ref_voice.wav", "/tmp/ref_voice.wav")
    except Exception as e:
        print(f"[bootstrap] No ref_voice.wav: {e}")

    # Run batch processor
    print("[bootstrap] Running batch processor...")
    subprocess.run(
        [sys.executable, "/tmp/batch.py", "--job-id", BID, "--work-dir", "/tmp", "--output-dir", "/tmp/output"],
        check=True,
    )

    # Upload outputs
    print("[bootstrap] Uploading outputs to R2...")
    for fp in glob.glob("/tmp/output/*"):
        r2_upload(fp, f"jobs/{BID}/output/{os.path.basename(fp)}")

    # Done marker
    R2.put_object(Bucket=BUCKET, Key=f"jobs/{BID}/output/done.json", Body=b'{"done":true}')
    print("[bootstrap] Done.")


if __name__ == "__main__":
    main()
