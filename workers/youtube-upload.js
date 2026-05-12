/**
 * Cloudflare Worker — Cloud-side YouTube upload from R2
 *
 * Setup:
 * 1. wrangler secret put YT_REFRESH_TOKEN   (from youtube_uploader.py --auth)
 * 2. wrangler secret put YT_CLIENT_ID       (from Google Cloud Console)
 * 3. wrangler secret put YT_CLIENT_SECRET   (from Google Cloud Console)
 * 4. wrangler secret put R2_ACCESS_KEY_ID
 * 5. wrangler secret put R2_SECRET_ACCESS_KEY
 * 6. Bind R2 bucket as "VIDEOS" in wrangler.toml
 *
 * POST /upload
 * Body JSON:
 *   {
 *     "r2_key": "jobs/job_abc/output/lecture.mp4",
 *     "title": "Week 1 - Intro",
 *     "description": "...",
 *     "privacy": "private",
 *     "callback_key": "jobs/job_abc/output/yt_result.json"
 *   }
 */

const YT_UPLOAD_ENDPOINT = "https://www.googleapis.com/upload/youtube/v3/videos";
const YT_PLAYLIST_ENDPOINT = "https://www.googleapis.com/youtube/v3/playlistItems";
const YT_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token";

async function getAccessToken(env) {
  const resp = await fetch(YT_TOKEN_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: env.YT_REFRESH_TOKEN,
      client_id: env.YT_CLIENT_ID,
      client_secret: env.YT_CLIENT_SECRET,
    }),
  });
  const data = await resp.json();
  if (!data.access_token) {
    throw new Error(`Token refresh failed: ${JSON.stringify(data)}`);
  }
  return data.access_token;
}

async function addToPlaylist(accessToken, videoId, playlistId) {
  const resp = await fetch(
    `${YT_PLAYLIST_ENDPOINT}?part=snippet`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        snippet: {
          playlistId: playlistId,
          resourceId: { kind: "youtube#video", videoId: videoId },
        },
      }),
    }
  );
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Playlist insert failed ${resp.status}: ${text}`);
  }
  return await resp.json();
}

async function uploadToYouTube(accessToken, videoBlob, metadata) {
  const initResp = await fetch(
    `${YT_UPLOAD_ENDPOINT}?uploadType=resumable&part=snippet,status`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Length": videoBlob.size.toString(),
        "X-Upload-Content-Type": "video/mp4",
      },
      body: JSON.stringify({
        snippet: {
          title: metadata.title,
          description: metadata.description || "",
          categoryId: "27",
        },
        status: {
          privacyStatus: metadata.privacy || "private",
          selfDeclaredMadeForKids: false,
        },
      }),
    }
  );

  if (!initResp.ok) {
    const text = await initResp.text();
    throw new Error(`YouTube init failed ${initResp.status}: ${text}`);
  }

  const uploadUrl = initResp.headers.get("Location");
  if (!uploadUrl) {
    throw new Error("No Location header from YouTube resumable init");
  }

  const uploadResp = await fetch(uploadUrl, {
    method: "PUT",
    headers: {
      "Content-Type": "video/mp4",
      "Content-Length": videoBlob.size.toString(),
    },
    body: videoBlob,
  });

  if (!uploadResp.ok) {
    const text = await uploadResp.text();
    throw new Error(`YouTube upload failed ${uploadResp.status}: ${text}`);
  }

  const result = await uploadResp.json();
  return {
    videoId: result.id,
    url: `https://youtube.com/watch?v=${result.id}`,
  };
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // ── List Playlists ──
    if (url.pathname === "/playlists" && request.method === "GET") {
      try {
        const accessToken = await getAccessToken(env);
        const resp = await fetch(
          `https://www.googleapis.com/youtube/v3/playlists?part=snippet&mine=true&maxResults=50`,
          { headers: { Authorization: `Bearer ${accessToken}` } }
        );
        const data = await resp.json();
        if (!resp.ok) {
          return new Response(JSON.stringify({ error: data }), { status: resp.status, headers: { "Content-Type": "application/json" } });
        }
        const playlists = (data.items || []).map((p) => ({
          id: p.id,
          title: p.snippet.title,
        }));
        return new Response(JSON.stringify({ playlists }), {
          headers: { "Content-Type": "application/json" },
        });
      } catch (e) {
        return new Response(JSON.stringify({ error: e.message }), {
          status: 500,
          headers: { "Content-Type": "application/json" },
        });
      }
    }

    // ── Upload Video ──
    if (url.pathname === "/upload" && request.method === "POST") {
      let body;
      try {
        body = await request.json();
      } catch (e) {
        return new Response(JSON.stringify({ error: "Invalid JSON" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }

      const { r2_key, title, description, privacy, callback_key, playlist_id } = body;
      if (!r2_key || !title) {
        return new Response(
          JSON.stringify({ error: "Missing r2_key or title" }),
          { status: 400, headers: { "Content-Type": "application/json" } }
        );
      }

      try {
        const accessToken = await getAccessToken(env);

        const obj = await env.VIDEOS.get(r2_key);
        if (!obj) {
          return new Response(
            JSON.stringify({ error: `R2 object not found: ${r2_key}` }),
            { status: 404, headers: { "Content-Type": "application/json" } }
          );
        }

        const videoBlob = await obj.blob();

        const ytResult = await uploadToYouTube(accessToken, videoBlob, {
          title,
          description: description || "",
          privacy: privacy || "private",
        });

        // Add to playlist if requested
        let playlistResult = null;
        if (playlist_id) {
          playlistResult = await addToPlaylist(accessToken, ytResult.videoId, playlist_id);
        }

        const resultPayload = JSON.stringify({
          success: true,
          videoId: ytResult.videoId,
          url: ytResult.url,
          playlistId: playlist_id || null,
          playlistItemId: playlistResult ? playlistResult.id : null,
          uploadedAt: new Date().toISOString(),
        });

        if (callback_key) {
          await env.VIDEOS.put(callback_key, resultPayload, {
            httpMetadata: { contentType: "application/json" },
          });
        }

        return new Response(resultPayload, {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      } catch (e) {
        const errorPayload = JSON.stringify({
          success: false,
          error: e.message,
        });
        if (callback_key) {
          await env.VIDEOS.put(callback_key, errorPayload, {
            httpMetadata: { contentType: "application/json" },
          });
        }
        return new Response(errorPayload, {
          status: 500,
          headers: { "Content-Type": "application/json" },
        });
      }
    }

    if (url.pathname === "/health") {
      return new Response(JSON.stringify({ ok: true }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response("Not found", { status: 404 });
  },
};
