"""CommentLens MCP server.

Exposes the hosted CommentLens API (fine-tuned BanglishBERT + news-matching index)
as MCP tools, so an MCP client such as Claude can analyse Bangla/Banglish comment
sections and check whether a viral claim has news support.

Transport: stdio (default).  API: POST /predict and POST /match_claim of the
commentlens-api service.  Set COMMENTLENS_API_URL to point at another deployment.
"""
import os
import time
from typing import Dict, List

import httpx
from mcp.server.fastmcp import FastMCP

API_URL = os.environ.get("COMMENTLENS_API_URL", "https://commentlens-api.onrender.com").rstrip("/")
CLASSES = ["opinion", "claim", "general", "spam-scam", "toxic"]
MAX_BATCH = 100

mcp = FastMCP("commentlens")


def _post(path: str, payload: dict, retries: int = 3) -> dict:
    """POST to the API. The free-tier server can be asleep, so allow a long first call."""
    last_error = None
    for attempt in range(retries):
        try:
            r = httpx.post(f"{API_URL}{path}", json=payload, timeout=90)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            last_error = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"CommentLens API call failed: {last_error}")


@mcp.tool()
def classify_comment(text: str) -> dict:
    """Classify one Bangla/Banglish comment into opinion, claim, general, spam-scam or toxic.

    Returns the label, the confidence and the score of every class.
    """
    res = _post("/predict", {"text": text[:2000]})
    return {"label": res["label"], "confidence": res["confidence"], "scores": res["scores"]}


@mcp.tool()
def analyze_comments(comments: List[str]) -> dict:
    """Analyse a batch of comments (up to 100) and return the opinion-share breakdown.

    Output: how many comments fall in each class, the share of each class in percent,
    and a simple Comment Health Score from 0 to 100 (100 minus toxic% minus half of
    spam-scam%, clamped). The score is a transparent heuristic, not a verdict.
    """
    comments = [c.strip() for c in comments if c and c.strip()][:MAX_BATCH]
    if not comments:
        raise ValueError("Provide at least one non-empty comment.")
    counts: Dict[str, int] = {c: 0 for c in CLASSES}
    claim_comments: List[str] = []
    for text in comments:
        label = _post("/predict", {"text": text[:2000]})["label"]
        counts[label] = counts.get(label, 0) + 1
        if label == "claim":
            claim_comments.append(text)
    total = len(comments)
    share = {c: round(counts[c] / total * 100, 1) for c in counts}
    health = max(0.0, min(100.0, 100 - share["toxic"] - 0.5 * share["spam-scam"]))
    return {
        "total": total,
        "counts": counts,
        "share_percent": share,
        "comment_health_score": round(health, 1),
        "claim_comments": claim_comments,
    }


@mcp.tool()
def check_claim(text: str, top_k: int = 3) -> dict:
    """Check whether a claim or headline text is supported by recent news coverage.

    Searches the daily-updated news index. 'backed' is true only for a strong match;
    false means no strong news match was found, which is a lead for a human to check
    and NOT a statement that the claim is false.
    """
    res = _post("/match_claim", {"text": text[:2000], "top_k": max(1, min(top_k, 20))})
    return {"backed": res.get("backed", False), "matches": res.get("matches", [])}


if __name__ == "__main__":
    mcp.run(transport="stdio")
