"""V2: reproduce Agrawal et al.'s published AUCs from their released scores (no GPU, no generation).

Usage (repo root):  python scripts/10_v2_reproduce_agrawal.py
Writes results/v2_agrawal_auc.csv. Targets and tolerance were pre-registered in
baseline_validation.md (commit 60bfdb5) before this script was first run.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr import agrawal  # noqa: E402
from aitr.metrics import agrawal_bootstrap_auc, point_auc  # noqa: E402

TOLERANCE = 0.005
METHODS = ["IQ", "DQ1", "DQ2", "DQ3", "DQ", "IQ+DQ"]
# arXiv:2305.18248v3, Fig. 5 (a-c) and Fig. 9 (a-c); see baseline_validation.md
PUBLISHED = {
    "gpt-4":            [0.877, 0.885, 0.842, 0.859, 0.915, 0.927],
    "gpt-3.5-turbo":    [0.782, 0.540, 0.610, 0.528, 0.624, 0.792],
    "llama-2-70b-chat": [0.658, 0.504, 0.631, 0.572, 0.658, 0.702],
    "text-davinci-003": [0.695, 0.555, 0.588, 0.534, 0.621, 0.710],
    "llama-2-13b-chat": [0.657, 0.500, 0.686, 0.546, 0.696, 0.723],
    "llama-2-7b-chat":  [0.517, 0.500, 0.574, 0.501, 0.573, 0.575],
}


def method_scores(df: pd.DataFrame) -> dict:
    """Agrawal orientation: score = 1 - released prob (higher = more likely hallucinated)."""
    s = {"IQ": 1 - df["iq_lm"], "DQ1": 1 - df["dq1"], "DQ2": 1 - df["dq2"], "DQ3": 1 - df["dq3"]}
    s["DQ"] = (s["DQ1"] + s["DQ2"] + s["DQ3"]) / 3
    s["IQ+DQ"] = (s["IQ"] + s["DQ"]) / 2
    return s


def evaluate(df: pd.DataFrame, label: str, published=None) -> list[dict]:
    rows = []
    for i, (method, score) in enumerate(method_scores(df).items()):
        boot = agrawal_bootstrap_auc(score.values, df["y"].values)
        row = {"model": label, "method": method, "n": len(df), "n_hallucinated": int(df["y"].sum()),
               "published": published[i] if published else None,
               "ours_bootstrap_mean": round(boot["auc_mean"], 4), "ours_point_auc": round(point_auc(score, df["y"]), 4),
               "flipped": boot["flipped"], "resamples_used": boot["n_resamples_used"]}
        if published:
            row["abs_diff"] = round(abs(boot["auc_mean"] - published[i]), 4)
            row["pass"] = row["abs_diff"] <= TOLERANCE
        rows.append(row)
    return rows


def main():
    raw = ROOT / "data/raw/agrawal"
    rows = []
    for model, published in PUBLISHED.items():
        rows += evaluate(agrawal.load_model(raw, model), model, published)
    # Sensitivity only (not part of the pass/fail check): 13B without the duplicated rows (D-014).
    dedup = agrawal.drop_duplicate_refs(agrawal.load_model(raw, "llama-2-13b-chat"))
    rows += evaluate(dedup, "llama-2-13b-chat [dedup]")

    out = pd.DataFrame(rows)
    path = ROOT / "results/v2_agrawal_auc.csv"
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(out.to_string(index=False))
    checked = out.dropna(subset=["published"])
    print(f"\nV2: {int(checked['pass'].sum())}/{len(checked)} cells within ±{TOLERANCE}; "
          f"max |diff| = {checked['abs_diff'].max():.4f}; written {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
