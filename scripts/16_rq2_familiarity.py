"""RQ2 on Agrawal et al.'s released data: is fabrication more frequent for less-published topics? (PC-005)

Usage (repo root):  python scripts/16_rq2_familiarity.py
Needs results/e1_scores_main.csv (E1) for the per-tertile judge AUROCs. OpenAlex counts are cached.
Outputs: results/rq2_topic_familiarity.csv, rq2_slopes.csv, rq2_tertiles.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr import agrawal, splits  # noqa: E402
from aitr.oracle import Oracle  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/protocol_v1.yaml").read_text())
SEED, NB = CFG["seeds"]["bootstrap"], CFG["metrics"]["bootstrap_resamples"]


def slope(x, y, extra=None):
    X = x.reshape(-1, 1) if extra is None else np.column_stack([x, extra])
    return LogisticRegression(C=1e6, max_iter=1000).fit(X, y).coef_[0, 0]


def boot_slope(df, extra_cols=None):
    groups = [np.flatnonzero(df["topic"].values == t) for t in df["topic"].unique()]
    x, y = df["fam"].values, df["y"].values
    extra = df[extra_cols].values if extra_cols else None
    rng = np.random.default_rng(SEED)
    b = []
    for _ in range(NB):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) == 2:
            b.append(slope(x[idx], y[idx], None if extra is None else extra[idx]))
    return slope(x, y, extra), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    topics = splits.topics("main")
    oracle = Oracle(ROOT / "results/raw/oracle_cache.jsonl")
    fam = pd.DataFrame({"topic": topics, "leaf": [t.split(":", 1)[-1].strip() for t in topics]})
    fam["openalex_count"] = [oracle.openalex_count(l) for l in fam["leaf"]]
    fam["fam"] = np.log10(fam["openalex_count"] + 1)
    fam["tertile"] = pd.qcut(fam["fam"].rank(method="first"), 3, labels=["low", "mid", "high"])
    fam.to_csv(ROOT / "results/rq2_topic_familiarity.csv", index=False)

    frames = []
    for m in agrawal.MODELS:
        d = agrawal.load_model(ROOT / "data/raw/agrawal", m)
        if m == "llama-2-13b-chat":
            d = agrawal.drop_duplicate_refs(d)
        frames.append(d[d["topic"].isin(set(topics))][["model", "row", "topic", "y"]])
    refs = pd.concat(frames, ignore_index=True).merge(fam[["topic", "fam", "tertile"]], on="topic")

    rows = []
    for m in agrawal.MODELS:
        s, lo, hi = boot_slope(refs[refs["model"] == m].reset_index(drop=True))
        rows.append({"model": m, "slope_per_log10": round(s, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "h2_supported": hi < 0})
    dummies = pd.get_dummies(refs["model"], drop_first=True, dtype=float)
    pooled = pd.concat([refs, dummies], axis=1)
    s, lo, hi = boot_slope(pooled, list(dummies.columns))
    rows.append({"model": "pooled (model fixed effects)", "slope_per_log10": round(s, 4), "ci_lo": round(lo, 4),
                 "ci_hi": round(hi, 4), "h2_supported": hi < 0})
    slopes = pd.DataFrame(rows)
    slopes.to_csv(ROOT / "results/rq2_slopes.csv", index=False)

    scores = pd.read_csv(ROOT / "results/e1_scores_main.csv").merge(fam[["topic", "tertile"]], on="topic")
    trows = []
    for (m, t), g in scores.groupby(["model", "tertile"], observed=True):
        r = {"model": m, "tertile": t, "n": len(g), "fabrication_rate": round(g["y"].mean(), 4)}
        for j in ("j_lm", "j_str", "j_nli"):
            r[f"auroc_{j}"] = round(roc_auc_score(g["y"], g[j]), 4) if g["y"].nunique() == 2 else None
        trows.append(r)
    tert = pd.DataFrame(trows)
    tert.to_csv(ROOT / "results/rq2_tertiles.csv", index=False)

    print(fam[["openalex_count"]].describe().round(0).T.to_string(), "\n")
    print(fam.groupby("tertile", observed=True)["openalex_count"].agg(["min", "median", "max"]).to_string(), "\n")
    print(slopes.to_string(index=False), "\n")
    with pd.option_context("display.width", 200):
        print(tert.to_string(index=False))


if __name__ == "__main__":
    main()
