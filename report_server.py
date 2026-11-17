"""CommentLens report MCP server.

Turns the output of the CommentLens analysis tools (analyze_comments or
analyze_youtube_video) into a readable Markdown report, and compares two
analyses side by side (for example two videos, or the same video on two days).

It makes no network calls and writes no files: it only formats the numbers it
is given, so it is safe to run next to any other MCP server.

Like the rest of CommentLens it never says a claim is true or false. A claim is
only reported as "backed by news" or "not backed by news".
"""
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

LABELS = ["toxic", "spam-scam", "claim", "opinion", "general"]
SMALL_SAMPLE = 30

mcp = FastMCP("commentlens-report")


def _check(analysis: dict, name: str) -> None:
    if not isinstance(analysis, dict):
        raise ValueError(f"{name} must be the JSON object returned by a CommentLens analysis tool.")
    missing = [k for k in ("total", "counts", "share_percent", "comment_health_score") if k not in analysis]
    if missing:
        raise ValueError(f"{name} is missing: {', '.join(missing)}. Pass the output of analyze_comments or analyze_youtube_video.")
    if not analysis["total"]:
        raise ValueError(f"{name} has no comments to report on.")


def _share(analysis: dict, label: str) -> float:
    return float(analysis["share_percent"].get(label, 0.0))


def _count(analysis: dict, label: str) -> int:
    return int(analysis["counts"].get(label, 0))


@mcp.tool()
def render_report(analysis: dict, title: str = "CommentLens report") -> dict:
    """Format one analysis as a Markdown report.

    Pass the JSON returned by analyze_comments or analyze_youtube_video. The report
    has the opinion-share table, the Comment Health Score with its formula, and, when
    present, the claim comments with whether news coverage backs them.
    """
    _check(analysis, "analysis")
    total = int(analysis["total"])
    lines: List[str] = [f"# {title[:120]}", ""]
    if analysis.get("video_id"):
        lines += [f"Video id: {analysis['video_id']}", ""]
    lines += [
        f"Comments analysed: **{total}**",
        f"Comment Health Score: **{analysis['comment_health_score']} / 100**",
        "(100 minus toxic share minus half of spam-scam share. A transparent heuristic, not a verdict.)",
        "",
        "| Class | Comments | Share |",
        "|---|---:|---:|",
    ]
    for label in LABELS:
        lines.append(f"| {label} | {_count(analysis, label)} | {_share(analysis, label)}% |")
    if total < SMALL_SAMPLE:
        lines += ["", f"Note: only {total} comments were analysed, so these shares can move a lot with a few more comments."]

    claims = analysis.get("claims_checked")
    if claims:
        lines += ["", "## Claims checked against news", ""]
        for item in claims:
            status = "backed by news" if item.get("backed_by_news") else "not backed by news"
            lines.append(f"- {str(item.get('claim', ''))[:200]} — **{status}**")
            for headline in (item.get("closest_headlines") or [])[:3]:
                if headline:
                    lines.append(f"  - closest headline: {headline}")
        lines += ["", "\"Not backed by news\" means no strong match was found. It is a lead for a human to check, not a statement that the claim is false."]
    elif analysis.get("claim_comments"):
        lines += ["", f"{len(analysis['claim_comments'])} comments were classified as claims. Run check_claim on them to see whether news coverage backs them."]
    return {"markdown": "\n".join(lines)}


@mcp.tool()
def compare_analyses(first: dict, second: dict, first_label: str = "A", second_label: str = "B") -> dict:
    """Compare two analyses (output of analyze_comments or analyze_youtube_video).

    Returns, for every class, the share in each analysis and the difference in
    percentage points, biggest change first, plus the Comment Health Score change
    and a Markdown table.
    """
    _check(first, "first")
    _check(second, "second")
    a_name, b_name = first_label[:40], second_label[:40]
    rows: List[Dict] = []
    for label in LABELS:
        a, b = _share(first, label), _share(second, label)
        rows.append({"class": label, a_name: a, b_name: b, "difference_points": round(b - a, 1)})
    rows.sort(key=lambda r: abs(r["difference_points"]), reverse=True)
    health_a, health_b = float(first["comment_health_score"]), float(second["comment_health_score"])

    md = [
        f"| Class | {a_name} share | {b_name} share | Change (points) |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['class']} | {r[a_name]}% | {r[b_name]}% | {r['difference_points']:+} |")
    md += [
        "",
        f"Comment Health Score: {a_name} {health_a}, {b_name} {health_b} ({round(health_b - health_a, 1):+}).",
        f"Comments analysed: {a_name} {int(first['total'])}, {b_name} {int(second['total'])}.",
    ]
    small = [n for n, d in ((a_name, first), (b_name, second)) if int(d["total"]) < SMALL_SAMPLE]
    if small:
        md.append(f"Note: {', '.join(small)} has fewer than {SMALL_SAMPLE} comments, so treat the differences as rough.")
    return {
        "differences": rows,
        "health_score_change": round(health_b - health_a, 1),
        "markdown": "\n".join(md),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
