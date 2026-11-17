"""CommentLens analyst agent, built as a LangGraph graph.

The agent works through a comment section in steps and uses the CommentLens MCP
servers in this repository as its tools (it is an MCP client):

    load -> classify -> (claims found?) -> verify -> report

  load      reads the comments: a text file with one comment per line, or the
            comments of a public YouTube video through the commentlens-youtube server
  classify  commentlens server, analyze_comments (class counts, opinion share, health score)
  verify    commentlens server, check_claim, for up to 5 distinct claim comments
  report    commentlens-report server, render_report (Markdown)

The routing after "classify" is a plain rule (go to "verify" only when claim
comments were found), not an LLM. The graph has no generative model in it, so it
runs without any API key, and a model could be added later as one more node.
Like the rest of CommentLens it never says a claim is true or false.

Usage:
    python agent/analyst_agent.py comments.txt
    python agent/analyst_agent.py --youtube https://youtu.be/VIDEO_ID --max 100 --out report.md
(the YouTube option needs YOUTUBE_API_KEY in the environment)
"""
import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
BATCH = 100          # analyze_comments takes up to 100 comments per call
MAX_CLAIMS = 5       # claims checked against news per run
LABELS = ["toxic", "spam-scam", "claim", "opinion", "general"]

_sessions: Dict[str, ClientSession] = {}


class AgentState(TypedDict, total=False):
    source: str
    max_comments: int
    comments: List[str]
    analysis: dict
    claims_checked: List[dict]
    report: str
    log: List[str]


async def _tool(server: str, name: str, args: dict) -> dict:
    result = await _sessions[server].call_tool(name, args)
    text = result.content[0].text if result.content else ""
    if result.isError:
        raise RuntimeError(f"{server}.{name} failed: {text[:200]}")
    return json.loads(text)


def _note(state: AgentState, message: str) -> List[str]:
    return state.get("log", []) + [message]


async def load(state: AgentState) -> AgentState:
    source = state["source"]
    if source.startswith("http") or source.startswith("youtu"):
        res = await _tool("youtube", "fetch_youtube_comments", {"video_url": source, "max_comments": state.get("max_comments", 100)})
        comments = res["comments"]
    else:
        comments = [line.strip() for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not comments:
        raise ValueError("No comments to analyse.")
    return {"comments": comments, "log": _note(state, f"load: {len(comments)} comments")}


async def classify(state: AgentState) -> AgentState:
    comments = state["comments"]
    counts = {label: 0 for label in LABELS}
    claims: List[str] = []
    for start in range(0, len(comments), BATCH):
        res = await _tool("commentlens", "analyze_comments", {"comments": comments[start:start + BATCH]})
        for label, n in res["counts"].items():
            counts[label] = counts.get(label, 0) + n
        claims += res.get("claim_comments", [])
    total = len(comments)
    share = {k: round(v / total * 100, 1) for k, v in counts.items()}
    health = max(0.0, min(100.0, 100 - share["toxic"] - 0.5 * share["spam-scam"]))
    analysis = {
        "total": total,
        "counts": counts,
        "share_percent": share,
        "comment_health_score": round(health, 1),
        "claim_comments": claims,
    }
    return {"analysis": analysis, "log": _note(state, f"classify: {counts['claim']} claim comments of {total}")}


def route_after_classify(state: AgentState) -> str:
    """The agent's one decision: only look for news support when there are claims."""
    return "verify" if state["analysis"]["claim_comments"] else "report"


async def verify(state: AgentState) -> AgentState:
    distinct: List[str] = []
    for text in state["analysis"]["claim_comments"]:
        if text not in distinct:
            distinct.append(text)
    checked = []
    for text in distinct[:MAX_CLAIMS]:
        res = await _tool("commentlens", "check_claim", {"text": text, "top_k": 3})
        checked.append(
            {
                "claim": text,
                "backed_by_news": bool(res.get("backed")),
                "closest_headlines": [m.get("headline") for m in res.get("matches", [])[:3]],
            }
        )
    return {"claims_checked": checked, "log": _note(state, f"verify: checked {len(checked)} distinct claims")}


async def report(state: AgentState) -> AgentState:
    analysis = dict(state["analysis"])
    if state.get("claims_checked"):
        analysis["claims_checked"] = state["claims_checked"]
    res = await _tool("report", "render_report", {"analysis": analysis, "title": "CommentLens analyst report"})
    return {"report": res["markdown"], "log": _note(state, "report: written")}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load", load)
    graph.add_node("classify", classify)
    graph.add_node("verify", verify)
    graph.add_node("report", report)
    graph.add_edge(START, "load")
    graph.add_edge("load", "classify")
    graph.add_conditional_edges("classify", route_after_classify, {"verify": "verify", "report": "report"})
    graph.add_edge("verify", "report")
    graph.add_edge("report", END)
    return graph.compile()


async def run(source: str, max_comments: int = 100) -> AgentState:
    servers = {
        "commentlens": "server.py",
        "youtube": "youtube_server.py",
        "report": "report_server.py",
    }
    async with AsyncExitStack() as stack:
        for name, script in servers.items():
            if name == "youtube" and not (source.startswith("http") or source.startswith("youtu")):
                continue  # only start the YouTube server when a video link was given
            params = StdioServerParameters(command=sys.executable, args=[str(ROOT / script)], env=dict(os.environ))
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            _sessions[name] = session
        return await build_graph().ainvoke({"source": source, "max_comments": max_comments})


def main() -> None:
    parser = argparse.ArgumentParser(description="CommentLens analyst agent")
    parser.add_argument("source", nargs="?", help="text file with one comment per line")
    parser.add_argument("--youtube", help="public YouTube video link (needs YOUTUBE_API_KEY)")
    parser.add_argument("--max", type=int, default=100, help="max YouTube comments (1-300)")
    parser.add_argument("--out", help="also save the report to this file")
    args = parser.parse_args()
    source = args.youtube or args.source
    if not source:
        parser.error("give a comments file or --youtube LINK")
    final = asyncio.run(run(source, args.max))
    print("\n".join(final["log"]), file=sys.stderr)
    print(final["report"])
    if args.out:
        Path(args.out).write_text(final["report"], encoding="utf-8")


if __name__ == "__main__":
    main()
