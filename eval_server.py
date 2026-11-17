"""CommentLens evaluation MCP server.

Measures how well the hosted CommentLens classifier does on labelled comments:
accuracy, precision, recall and F1 for every class, macro-F1, and the confusion
matrix. It sends the comments to the CommentLens API (POST /predict) and compares
the answers with the labels you provide. It never changes the model or the API.

Files are read with the same folder rule as the dataset server: only .csv files
inside COMMENTLENS_DATA_DIR. Run it on a held-out test set, not on training data.
"""
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import httpx
from mcp.server.fastmcp import FastMCP

from dataset_server import _need_column, _read_rows

API_URL = os.environ.get("COMMENTLENS_API_URL", "https://commentlens-api.onrender.com").rstrip("/")
CLASSES = ["toxic", "spam-scam", "claim", "opinion", "general"]
MAX_ROWS = 500
MAX_INLINE = 100

mcp = FastMCP("commentlens-eval")


def _predict(client: httpx.Client, text: str) -> str:
    """Ask the API for one label. The free-tier host sleeps, so retry a few times."""
    last = None
    for attempt in range(3):
        try:
            resp = client.post(f"{API_URL}/predict", json={"text": text[:2000]})
            resp.raise_for_status()
            return resp.json()["label"]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"CommentLens API call failed after retries: {type(last).__name__}")


def metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """Accuracy, per-class precision/recall/F1, macro-F1 and the confusion matrix."""
    labels = CLASSES + sorted((set(y_true) | set(y_pred)) - set(CLASSES))
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    per_class = {}
    for c in labels:
        tp = matrix[c][c]
        fp = sum(matrix[t][c] for t in labels if t != c)
        fn = sum(matrix[c][p] for p in labels if p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    present = [c for c in labels if per_class[c]["support"] > 0]
    macro_f1 = sum(per_class[c]["f1"] for c in present) / len(present)
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_f1_note": "Average F1 over the classes that appear in the labels.",
        "per_class": per_class,
        "confusion_matrix": {"rows_are_true_labels": True, "matrix": matrix},
    }


def _run(texts: List[str], labels: List[str]) -> Dict:
    with httpx.Client(timeout=90) as client:
        with ThreadPoolExecutor(max_workers=6) as pool:
            predicted = list(pool.map(lambda t: _predict(client, t), texts))
    return metrics(labels, predicted)


@mcp.tool()
def evaluate_csv(csv_file: str, text_col: str = "text", label_col: str = "label", max_rows: int = 200, seed: int = 42) -> dict:
    """Score the live classifier on a labelled CSV (a random sample of up to 500 rows).

    csv_file is a file name inside the data folder. Rows whose label is not one of
    toxic, spam-scam, claim, opinion, general are skipped and counted. Use a held-out
    test set, not the training file.
    """
    rows = _read_rows(csv_file)
    _need_column(rows, text_col, csv_file)
    _need_column(rows, label_col, csv_file)
    usable = [(r[text_col].strip(), r[label_col].strip()) for r in rows if (r.get(text_col) or "").strip() and (r.get(label_col) or "").strip() in CLASSES]
    skipped = len(rows) - len(usable)
    if not usable:
        raise ValueError("No usable rows: every row had empty text or a label outside the 5-class scheme.")
    limit = max(1, min(MAX_ROWS, int(max_rows)))
    sample = random.Random(seed).sample(usable, min(limit, len(usable)))
    result = _run([t for t, _ in sample], [l for _, l in sample])
    result.update(
        {
            "file": csv_file,
            "rows_in_file": len(rows),
            "rows_evaluated": len(sample),
            "rows_skipped": skipped,
            "note": "Computed on a random sample" if len(sample) < len(usable) else "Computed on every usable row",
        }
    )
    return result


@mcp.tool()
def evaluate_pairs(texts: List[str], labels: List[str]) -> dict:
    """Score the live classifier on a few comments given directly (up to 100).

    texts and labels must have the same length; labels use the 5-class scheme.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length.")
    if not texts or len(texts) > MAX_INLINE:
        raise ValueError(f"Give between 1 and {MAX_INLINE} comments.")
    bad = sorted({l for l in labels if l not in CLASSES})
    if bad:
        raise ValueError(f"Unknown labels: {', '.join(bad)}. Use: {', '.join(CLASSES)}.")
    result = _run([t.strip() for t in texts], list(labels))
    result["rows_evaluated"] = len(texts)
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
