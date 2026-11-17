# CommentLens MCP servers

Five MCP (Model Context Protocol) servers and a small agent that let an AI client such as Claude analyse Bangla/Banglish comment sections with the CommentLens engine. Three of the servers call the hosted CommentLens API ([commentlens-api](https://github.com/gulamsakaria/commentlens-api)): a fine-tuned BanglishBERT classifier and a daily-updated news-matching index (TF-IDF over headlines). The other two only work on data you give them.

Transport: stdio. Built for the International AI Builders Congress 2026.

## The servers

| Server | Tools | Needs |
|---|---|---|
| server.py (commentlens) | classify_comment, analyze_comments, check_claim | the hosted API |
| youtube_server.py (commentlens-youtube) | fetch_youtube_comments, analyze_youtube_video | YOUTUBE_API_KEY, the hosted API |
| report_server.py (commentlens-report) | render_report, compare_analyses | nothing (no network) |
| dataset_server.py (commentlens-dataset) | label_stats, sample_comments, annotator_agreement | a data folder (COMMENTLENS_DATA_DIR) |
| eval_server.py (commentlens-eval) | evaluate_csv, evaluate_pairs | the hosted API, a data folder |

Design rule for all of them: nothing says a claim is true or false. `backed: false` or "not backed by news" only means no strong news match was found, which is a lead for a human to check.

The Comment Health Score is a simple, transparent heuristic: 100 minus the toxic share minus half of the spam-scam share, clamped to 0-100.

## Run and register

```bash
pip install -r requirements.txt
python server.py

claude mcp add commentlens -- python /path/to/server.py
claude mcp add commentlens-youtube -- python /path/to/youtube_server.py
claude mcp add commentlens-report -- python /path/to/report_server.py
claude mcp add commentlens-dataset -e COMMENTLENS_DATA_DIR=/path/to/data -- python /path/to/dataset_server.py
claude mcp add commentlens-eval -e COMMENTLENS_DATA_DIR=/path/to/data -- python /path/to/eval_server.py
```

By default the API-backed servers call `https://commentlens-api.onrender.com`. Set `COMMENTLENS_API_URL` to use another deployment. The API runs on a free tier, so the first call after a quiet period can take 30-50 seconds while the server wakes up; the clients retry automatically.

## server.py: classify and check claims

| Tool | What it does |
|---|---|
| classify_comment(text) | Classifies one comment as opinion, claim, general, spam-scam or toxic, with confidence and all class scores. |
| analyze_comments(comments) | Takes up to 100 comments and returns class counts, the opinion-share breakdown in percent, the Comment Health Score and the list of claim comments. |
| check_claim(text, top_k) | Searches the news index and says whether the claim has strong news support (`backed`), with the closest headlines. |

## youtube_server.py: comments of a YouTube video

| Tool | What it does |
|---|---|
| fetch_youtube_comments | Top-level comments of a public video (most relevant first), up to 300. |
| analyze_youtube_video | Fetches the comments, classifies each one, and returns class counts, opinion share, the Comment Health Score and up to 5 distinct claim comments checked against the news archive. |

It needs a YouTube Data API v3 key in the YOUTUBE_API_KEY environment variable of the server process. The key is sent in a request header and is never returned by a tool.

## report_server.py: reports and comparisons

| Tool | What it does |
|---|---|
| render_report(analysis, title) | Turns the output of analyze_comments or analyze_youtube_video into a Markdown report: opinion-share table, health score with its formula, and the claims checked against news. |
| compare_analyses(first, second, labels) | Compares two analyses (two videos, or one video on two days): share per class, change in points, health score change. Warns when a sample has fewer than 30 comments. |

No network calls and no files written; it only formats the numbers it is given.

## dataset_server.py: labelled data and annotator agreement

| Tool | What it does |
|---|---|
| label_stats(csv_file) | Row count and the count and share of every label in a labelled CSV. |
| sample_comments(csv_file, label, n) | Up to 20 random comments with a given label, for spot checks. |
| annotator_agreement(csv_files) | For 2 to 6 annotators (one CSV each): Cohen's kappa and percent agreement for every pair, Fleiss' kappa for three or more, and the label pairs they disagree on most. |

It reads only .csv files inside COMMENTLENS_DATA_DIR (default ./data) and refuses any path outside that folder. Nothing is uploaded or written. It exists because several team members label the same comments and we need to know how consistent they are.

## eval_server.py: how good is the classifier

| Tool | What it does |
|---|---|
| evaluate_csv(csv_file, max_rows) | Sends a random sample (up to 500 rows) of a labelled CSV to the live API and returns accuracy, precision, recall and F1 per class, macro-F1 and the confusion matrix. |
| evaluate_pairs(texts, labels) | The same for up to 100 comments given directly. |

Use a held-out test set, not the training file. It never changes the model or the API.

## agent/analyst_agent.py: an agent that uses the servers

A small LangGraph graph (load, classify, verify, report) that is an MCP client of the servers above: it reads a comment file or a YouTube video, classifies the comments, looks for news support only when claim comments were found, and writes a Markdown report with render_report. The routing is a plain rule, not an LLM, so it needs no model API key.

```bash
pip install -r agent/requirements.txt
python agent/analyst_agent.py comments.txt
python agent/analyst_agent.py --youtube https://youtu.be/VIDEO_ID --max 100 --out report.md
```

## Permissions

- server.py and eval_server.py send the comment text you pass in (or the sampled CSV rows) to the CommentLens API. No credentials.
- youtube_server.py also calls the YouTube Data API with your key. No files are read or written.
- report_server.py: no network, no files.
- dataset_server.py: reads .csv files inside one folder only. No network, nothing written.

## How it was checked

- All five servers connect to Claude Code (`claude mcp list` reports them as connected).
- Every server was called through a Python MCP stdio client. The API-backed tools were run against the live API, and the dataset server refused a path outside its folder.
- The kappa values of the dataset server match scikit-learn (Cohen's) and statsmodels (Fleiss') on test data, and the accuracy, per-class and macro-F1 numbers of the eval server match scikit-learn.
- The agent was run end to end against the live API on a small text file, once with claim comments (verify step used) and once without (verify step skipped).
- The YouTube server was tested with the YouTube API mocked and the live CommentLens API (link parsing, the key staying in the header, claim checking, the missing-key error). It has not been run against the real YouTube API yet.
- These checks were run as scratch scripts and are not part of the repository yet.
