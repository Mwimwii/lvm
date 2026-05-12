# Cloudflare Worker — YouTube Upload from R2

Zero-local-bandwidth YouTube upload. The Lightning L40S job renders your video, uploads it to R2, then calls this Worker to stream it from R2 → YouTube. Your machine never downloads the video.

## Setup

### 1. Get YouTube OAuth Refresh Token

Run locally (one-time):

```bash
python youtube_uploader.py --auth
# Follow browser prompt, authorize
# token.json is created
```

Extract the `refresh_token` field from `token.json`.

### 2. Create Cloudflare Worker

```bash
cd workers
npm install -g wrangler  # if not installed
wrangler login

# Set secrets
wrangler secret put YT_REFRESH_TOKEN    # paste refresh_token
wrangler secret put YT_CLIENT_ID        # from Google Cloud Console
wrangler secret put YT_CLIENT_SECRET   # from Google Cloud Console
wrangler secret put R2_ACCESS_KEY_ID    # your R2 access key
wrangler secret put R2_SECRET_ACCESS_KEY # your R2 secret key

# Update wrangler.toml bucket_name to your actual R2 bucket
# Deploy
wrangler deploy
```

### 3. Configure Client

Add to `.env`:

```bash
CF_WORKER_URL=https://lvm-youtube-uploader.YOUR_ACCOUNT.workers.dev/upload
```

Or enter it in the UI: **YouTube Upload (Cloudflare Worker)** accordion.

## How It Works

```
Client → R2 (script, pdf, manifest with yt metadata)
  ↓
Lightning L40S → render video → upload MP4 to R2
  ↓
L40S POSTs to CF Worker: {r2_key, title, description, privacy}
  ↓
CF Worker → reads MP4 from R2 → resumable upload to YouTube
  ↓
CF Worker → writes yt_result.json to R2
  ↓
Client polls yt_result.json → shows YouTube link
```

## Limitations

- Free Workers plan: 50MB body limit. Large videos need paid Workers ($5/mo = 500MB).
- Resumable upload is used, so the Worker streams the file. For very large videos (>500MB), consider splitting or using the local `youtube_uploader.py` path.
