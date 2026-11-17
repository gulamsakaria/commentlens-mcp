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

## Second server: YouTube comments (youtube_server.py)

Pulls the comments of a public YouTube video and analyses them with the hosted CommentLens API, so a whole comment section can be checked from a chat or coding session.

| Tool | What it does |
|---|---|
| fetch_youtube_comments | Top-level comments of a public video (most relevant first), up to 300. |
| analyze_youtube_video | Fetches the comments, classifies each one, and returns class counts, opinion share, the Comment Health Score and up to 5 distinct claim comments checked against the news archive. |

It needs a YouTube Data API v3 key in the YOUTUBE_API_KEY environment variable of the server process. The key is sent in a request header and is never returned by a tool. Claims are only reported as backed or not backed by news, never as true or false.

Run it with: python youtube_server.py

Register it with Claude Code: claude mcp add commentlens-youtube -- python /path/to/youtube_server.py

## How it was checked

- Both servers connect to Claude Code (claude mcp list reports them as connected).
- The main server was called through a Python MCP stdio client against the live API.
- The YouTube server was tested with the YouTube API mocked and the live CommentLens API, covering link parsing, the API key staying in the header, claim checking and the missing-key error. It has not been run against the real YouTube API yet.
