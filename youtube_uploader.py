"""
YouTube upload helper using Data API v3.

Setup:
1. Go to https://console.cloud.google.com/ → APIs & Services → Credentials
2. Create OAuth 2.0 Client ID (Desktop app)
3. Download JSON → save as client_secret.json in this folder
4. Enable "YouTube Data API v3" in the project
5. Run once: python youtube_uploader.py --auth   (opens browser)
6. token.json is saved — reuse for uploads

Usage:
    from youtube_uploader import upload_video
    upload_video("output.mp4", title="My Lecture", description="...")
"""
import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/youtube"]
CLIENT_SECRETS_FILE = "client_secret.json"
TOKEN_FILE = "token.json"


def _get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CLIENT_SECRETS_FILE):
                raise FileNotFoundError(
                    f"{CLIENT_SECRETS_FILE} not found. "
                    "Download it from Google Cloud Console → Credentials → OAuth 2.0 Desktop."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return creds


def upload_video(file_path, title, description="", tags=None, category_id="27", privacy="private"):
    """
    Upload a video to YouTube.

    privacy: 'private', 'unlisted', or 'public'
    category_id: 27 = Education, 22 = People & Blogs, etc.
    """
    creds = _get_credentials()
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags or [],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")

    video_id = response["id"]
    url = f"https://youtube.com/watch?v={video_id}"
    print(f"Upload complete: {url}")
    return {"video_id": video_id, "url": url}


def check_auth():
    """Verify credentials are valid. Returns True/False."""
    try:
        _get_credentials()
        return True
    except Exception as e:
        print(f"Auth check failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", action="store_true", help="Run OAuth flow once")
    parser.add_argument("--file", help="Video file to upload")
    parser.add_argument("--title", default="Uploaded Video")
    parser.add_argument("--description", default="")
    parser.add_argument("--privacy", default="private", choices=["private", "unlisted", "public"])
    args = parser.parse_args()

    if args.auth:
        print("Authenticating with YouTube...")
        _get_credentials()
        print(f"Saved to {TOKEN_FILE}. Ready to upload.")
    elif args.file:
        upload_video(args.file, args.title, args.description, privacy=args.privacy)
    else:
        parser.print_help()
