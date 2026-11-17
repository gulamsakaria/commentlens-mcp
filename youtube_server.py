"""CommentLens YouTube MCP server.

Lets an AI client (Claude, Claude Code, any MCP client) pull the comments of a
public YouTube video and run them through the hosted CommentLens API, so the
comment section can be analysed from inside a chat or coding session.

Needs a YouTube Data API v3 key in the YOUTUBE_API_KEY environment variable.
The key is sent in a request header and is never returned by any tool.

Like the main CommentLens server, it never says a claim is true or false: it
only reports whether recent news coverage supports it.
"""
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from urllib.parse import parse_qs, urlparse

import httpx
from mcp.server.fastmcp import FastMCP

API_URL = os.environ.get("COMMENTLENS_API_URL", "https://commentlens-api.onrender.com").rstrip("/")
YOUTUBE_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
LABELS = ["toxic", "spam-scam", "claim", "opinion", "general"]
MAX_COMMENTS = 300

mcp = FastMCP("commentlens-youtube")


def _client() -> httpx.Client:
    return httpx.Client(timeout=90)


def _video_id(video_url: str) -> str:
    """Pull the 11-character video id out of a YouTube link (or accept a bare id)."""
    text = video_url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", text):
        return text
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    video_id = None
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in ("youtube.com", "music.youtube.com"):
        if parsed.path == "/watch":
            video_id = (parse_qs(parsed.query).get("v") or [None])[0]
        else:
            m = re.match(r"^/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})", parsed.path)
            video_id = m.group(1) if m else None
    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Could not find a YouTube video id in that link.")
    return video_id


def _fetch_comments(client: httpx.Client, video_id: str, max_comments: int) -> List[str]:
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("YOUTUBE_API_KEY is not set in the environment of this MCP server.")
    comments: List[str] = []
    page_token = None
    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "order": "relevance",
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token
        resp = client.get(YOUTUBE_URL, params=params, headers={"x-goog-api-key": key})
        if resp.status_code != 200:
            reason = ""
            try:
                reason = resp.json()["error"]["errors"][0]["reason"]
            except Exception:
                pass
            if reason == "commentsDisabled":
                raise ValueError("Comments are turned off for this video.")
            raise RuntimeError(f"YouTube API returned HTTP {resp.status_code} ({reason or 'no reason given'}).")
        data = resp.json()
        for item in data.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"].get("textDisplay", "").strip()
            if text:
                comments.append(text[:2000])
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return comments[:max_comments]


def _api_post(client: httpx.Client, path: str, payload: dict) -> dict:
    """POST to the CommentLens API; retry because the free-tier host sleeps."""
    last = None
    for attempt in range(3):
        try:
            resp = client.post(f"{API_URL}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"CommentLens API call failed after retries: {type(last).__name__}")


@mcp.tool()
def fetch_youtube_comments(video_url: str, max_comments: int = 100) -> dict:
    """Fetch the top-level comments of a public YouTube video (most relevant first).

    Returns the video id, how many comments were fetched, and the comment texts.
    max_comments is limited to 1-300.
    """
    limit = max(1, min(MAX_COMMENTS, int(max_comments)))
    video_id = _video_id(video_url)
    with _client() as client:
        comments = _fetch_comments(client, video_id, limit)
    return {"video_id": video_id, "count": len(comments), "comments": comments}


@mcp.tool()
def analyze_youtube_video(video_url: str, max_comments: int = 100) -> dict:
    """Fetch a video's comments and classify them with CommentLens.

    Returns class counts, opinion share in percent, the Comment Health Score
    (100 - toxic% - 0.5 x spam-scam%, clamped to 0-100), and up to 5 distinct
    claim comments checked against the news archive. A claim is only reported as
    "backed" or "not backed by news"; it is never called true or false.
    """
    limit = max(1, min(MAX_COMMENTS, int(max_comments)))
    video_id = _video_id(video_url)
    with _client() as client:
        comments = _fetch_comments(client, video_id, limit)
        if not comments:
            return {"video_id": video_id, "total": 0, "note": "No comments found."}

        def classify(text: str) -> dict:
            return _api_post(client, "/predict", {"text": text})

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(classify, comments))

        counts = {label: 0 for label in LABELS}
        claim_texts: List[str] = []
        for text, res in zip(comments, results):
            label = res.get("label")
            if label in counts:
                counts[label] += 1
                if label == "claim" and text not in claim_texts:
                    claim_texts.append(text)

        total = len(results)
        share = {k: round(v / total * 100, 1) for k, v in counts.items()}
        health = max(0.0, min(100.0, 100 - share["toxic"] - 0.5 * share["spam-scam"]))

        claims_checked = []
        for text in claim_texts[:5]:
            match = _api_post(client, "/match_claim", {"text": text, "top_k": 3})
            claims_checked.append(
                {
                    "claim": text,
                    "backed_by_news": bool(match.get("backed")),
                    "closest_headlines": [m.get("headline") for m in match.get("matches", [])[:3]],
                }
            )

    return {
        "video_id": video_id,
        "total": total,
        "counts": counts,
        "share_percent": share,
        "comment_health_score": round(health, 1),
        "claims_checked": claims_checked,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
