"""CommentLens dataset MCP server.

Read-only access to the team's labelled comment files (CSV), including a
check of how much several annotators agree with each other (Cohen's kappa for
each pair, Fleiss' kappa for three or more).

It only reads .csv files inside one folder, set with COMMENTLENS_DATA_DIR
(default: ./data). Paths outside that folder are refused. Nothing is uploaded
and nothing is written.
"""
import csv
import os
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

DATA_DIR = Path(os.environ.get("COMMENTLENS_DATA_DIR", "data")).resolve()
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_SAMPLE = 20

mcp = FastMCP("commentlens-dataset")


def _resolve(name: str) -> Path:
    """Turn a file name into a path, only if it is a .csv inside the data folder."""
    path = (DATA_DIR / name).resolve()
    if not path.is_relative_to(DATA_DIR):
        raise ValueError("The file must be inside the data folder (COMMENTLENS_DATA_DIR).")
    if path.suffix.lower() != ".csv":
        raise ValueError("Only .csv files are supported.")
    if not path.is_file():
        raise FileNotFoundError(f"No such file in the data folder: {name}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("That file is larger than 100 MB.")
    return path


def _read_rows(name: str) -> List[Dict[str, str]]:
    with _resolve(name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _need_column(rows: List[Dict[str, str]], column: str, file_name: str) -> None:
    if not rows:
        raise ValueError(f"{file_name} has no rows.")
    if column not in rows[0]:
        raise ValueError(f"{file_name} has no '{column}' column. Columns: {', '.join(rows[0].keys())}")


def cohen_kappa(a: List[str], b: List[str]) -> Dict[str, float]:
    """Cohen's kappa and plain percent agreement for two equal-length label lists."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if expected == 1:
        kappa = 1.0 if observed == 1 else 0.0
    else:
        kappa = (observed - expected) / (1 - expected)
    return {"kappa": round(kappa, 4), "percent_agreement": round(observed * 100, 2)}


def fleiss_kappa(ratings: List[List[str]]) -> float:
    """Fleiss' kappa. Every item must have the same number of raters."""
    n_items, n_raters = len(ratings), len(ratings[0])
    categories = sorted({label for row in ratings for label in row})
    totals = {c: 0 for c in categories}
    p_items = 0.0
    for row in ratings:
        counts = Counter(row)
        for c in categories:
            totals[c] += counts[c]
        p_items += (sum(v * v for v in counts.values()) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_items / n_items
    p_e = sum((totals[c] / (n_items * n_raters)) ** 2 for c in categories)
    if p_e == 1:
        return 1.0 if p_bar == 1 else 0.0
    return round((p_bar - p_e) / (1 - p_e), 4)


def _landis_koch(kappa: float) -> str:
    """Conventional reading of a kappa value (Landis and Koch, 1977)."""
    if kappa < 0:
        return "poor (worse than chance)"
    for upper, name in ((0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if kappa <= upper:
            return name
    return "almost perfect"


@mcp.tool()
def label_stats(csv_file: str, label_col: str = "label") -> dict:
    """Count how many comments each label has in a labelled CSV file.

    Returns the row count, the count and share of every label, and how many rows
    have no label. csv_file is a file name inside the data folder.
    """
    rows = _read_rows(csv_file)
    _need_column(rows, label_col, csv_file)
    labels = [(r.get(label_col) or "").strip() for r in rows]
    counts = Counter(l for l in labels if l)
    labelled = sum(counts.values())
    return {
        "file": csv_file,
        "rows": len(rows),
        "unlabelled_rows": len(rows) - labelled,
        "labels": {k: {"count": v, "percent": round(v / labelled * 100, 2)} for k, v in counts.most_common()},
    }


@mcp.tool()
def sample_comments(csv_file: str, label: str, n: int = 5, label_col: str = "label", text_col: str = "text", seed: int = 42) -> dict:
    """Return up to 20 random comments that carry a given label (for spot checks)."""
    rows = _read_rows(csv_file)
    _need_column(rows, label_col, csv_file)
    _need_column(rows, text_col, csv_file)
    pool = [r[text_col].strip() for r in rows if (r.get(label_col) or "").strip() == label and (r.get(text_col) or "").strip()]
    rng = random.Random(seed)
    picked = rng.sample(pool, min(max(1, min(n, MAX_SAMPLE)), len(pool)))
    return {"label": label, "available": len(pool), "comments": [t[:300] for t in picked]}


@mcp.tool()
def annotator_agreement(csv_files: List[str], key_col: str = "id", label_col: str = "label") -> dict:
    """Measure how much annotators agree, one CSV file per annotator (2 to 6 files).

    Rows are matched on key_col (falls back to a 'text' column when there is no id
    column). Only comments labelled by every annotator are compared. Returns Cohen's
    kappa and percent agreement for each pair, Fleiss' kappa when there are three or
    more annotators, and the label pairs annotators disagree on most.
    """
    if not 2 <= len(csv_files) <= 6:
        raise ValueError("Give between 2 and 6 CSV files, one per annotator.")
    labelled: List[Dict[str, str]] = []
    for name in csv_files:
        rows = _read_rows(name)
        key = key_col if rows and key_col in rows[0] else "text"
        _need_column(rows, key, name)
        _need_column(rows, label_col, name)
        mapping: Dict[str, str] = {}
        for r in rows:
            k, lab = (r.get(key) or "").strip(), (r.get(label_col) or "").strip()
            if k and lab and k not in mapping:
                mapping[k] = lab
        labelled.append(mapping)

    common = sorted(set.intersection(*(set(m) for m in labelled)))
    if not common:
        raise ValueError("No comment was labelled by every annotator. Check key_col and that the files cover the same comments.")
    series = [[m[k] for k in common] for m in labelled]

    pairs = []
    for i, j in combinations(range(len(csv_files)), 2):
        res = cohen_kappa(series[i], series[j])
        pairs.append({"annotators": [csv_files[i], csv_files[j]], **res})
    mean_kappa = round(sum(p["kappa"] for p in pairs) / len(pairs), 4)

    confusions: Counter = Counter()
    for i, j in combinations(range(len(csv_files)), 2):
        for x, y in zip(series[i], series[j]):
            if x != y:
                confusions[" vs ".join(sorted((x, y)))] += 1

    result = {
        "comments_compared": len(common),
        "per_file_labelled": {n: len(m) for n, m in zip(csv_files, labelled)},
        "pairwise": pairs,
        "mean_pairwise_kappa": mean_kappa,
        "reading": _landis_koch(mean_kappa),
        "reading_note": "Conventional Landis and Koch (1977) bands; a rule of thumb, not a test.",
        "most_disagreed_label_pairs": [{"labels": k, "count": v} for k, v in confusions.most_common(5)],
    }
    if len(csv_files) >= 3:
        result["fleiss_kappa"] = fleiss_kappa([list(row) for row in zip(*series)])
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
