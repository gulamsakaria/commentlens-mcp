# CommentLens MCP server

An MCP (Model Context Protocol) server that lets an AI client such as Claude analyse Bangla/Banglish comment sections with the CommentLens engine. It is a thin wrapper over the hosted CommentLens API ([commentlens-api](https://github.com/gulamsakaria/commentlens-api)): a fine-tuned BanglishBERT classifier and a daily-updated news-matching index.

Transport: stdio. Built for the International AI Builders Congress 2026.

## Tools

| Tool | What it does |
| --- | --- |
| classify_comment(text) | Classifies one comment as opinion, claim, general, spam-scam or toxic, with confidence and all class scores. |
| analyze_comments(comments) | Takes up to 100 comments and returns class counts, the opinion-share breakdown in percent, a Comment Health Score (0-100) and the list of claim comments. |
| check_claim(text, top_k) | Searches the news index and says whether the claim has strong news support (backed), with the closest headlines and links. |

Design rule: the server never says a claim is true or false. backed: false only means no strong news match was found, which is a lead for a human to check.

The Comment Health Score is a simple, transparent heuristic: 100 minus the toxic share minus half of the spam-scam share, clamped to 0-100.

## Run

    pip install -r requirements.txt
    python server.py

Add it to Claude Code:

    claude mcp add commentlens -- python /path/to/server.py

By default it calls https://commentlens-api.onrender.com. Set COMMENTLENS_API_URL to use another deployment. The API runs on a free tier, so the first call after a quiet period can take 30-50 seconds while the server wakes up; the client retries automatically.

## Permissions

The server only sends the comment or claim text you pass in to the CommentLens API. It reads no files, stores nothing and needs no credentials.
