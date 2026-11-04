"""E2 step 2: label references with the oracle and score detectors D1-D6 (protocol §3, §6, §7).

Usage (repo root):
    python scripts/15_e2_score.py --split pilot      # pipeline check: no AUROC printed
    python scripts/15_e2_score.py --split e2         # RQ1-RQ3 results
Reads results/raw/e2_<split>/*.json from scripts/14_e2_generate.py. NLI runs on CPU by default:
on the Apple GPU, long premises were ~4 s/pair vs 0.7 s/pair on CPU (measured 2026-11-04).
Outputs: results/e2_refs_<split>.csv; for e2 also e2_auroc.csv, e2_deltas.csv, e2_familiarity.csv,
e2_n_sensitivity.csv, e2_cost.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr import judges, prompts  # noqa: E402
from aitr.metrics import cluster_bootstrap_auc, paired_delta  # noqa: E402
from aitr.oracle import Oracle  # noqa: E402
from aitr.seed import set_seed  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/protocol_v1.yaml").read_text())
DETECTORS = ["d1_nli", "d2_unigram", "d3_logprob", "d4_verbal", "d5_dq", "d6_iq", "j_nli", "j_str"]


def load(split: str) -> tuple[pd.DataFrame, list]:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / f"results/raw/e2_{split}").glob("*.json"))]
    rows = []
    for t in recs:
        for r in t["refs"]:
            rows.append({"topic": t["topic"], "topic_index": t["topic_index"], "pos": r["pos"], "title": r["title"],
                         "parse_method": t["parse_method"], "nll": r["nll"],
                         "dq1": np.mean([prompts.dq_yes(a) for a in r["dq1"]]),
                         "dq2": np.mean([prompts.dq_yes(a) for a in r["dq2"]]),
                         "dq3": np.mean([prompts.dq_yes(a) for a in r["dq3"]]),
                         "iq_lm": float(np.mean(r["iq_frac"])), "verbal_p": r["verbal_p"],
                         "iq": r["iq"]})
    return pd.DataFrame(rows), recs


def score(df: pd.DataFrame, recs: list, device: str, nli_batch: int) -> pd.DataFrame:
    from selfcheckgpt.modeling_selfcheck import SelfCheckNgram
    samples = {t["topic_index"]: t["samples"] for t in recs}
    mains = {t["topic_index"]: t["main_answer"] for t in recs}
    scorer = judges.DebertaContradiction(CFG["baseline"]["nli_model"], device, CFG["baseline"]["nli_max_length"], nli_batch)
    uni = SelfCheckNgram(n=1)
    nli_per_sample, d2, jn, js = [], [], [], []
    for k, r in df.iterrows():
        s = samples[r["topic_index"]]
        # D1: title vs each sampled answer, encoded (hypothesis=title, premise=sample) as SelfCheckNLI
        nli_per_sample.append(scorer(s, [r["title"]] * len(s)).tolist())
        d2.append(uni.predict(sentences=[r["title"]], passage=mains[r["topic_index"]], sampled_passages=s)
                  ["sent_level"]["max_neg_logprob"][0])
        rule, _ = judges.j_nli([" ".join(a.split()) for a in r["iq"]], scorer)
        jn.append(rule)
        js.append(judges.j_str(r["iq"]))
        if k % 50 == 0:
            print(f"  scored {k + 1}/{len(df)}", flush=True)
    df["nli_per_sample"] = nli_per_sample
    df["d1_nli"] = [float(np.mean(x)) for x in nli_per_sample]
    df["d2_unigram"] = d2
    df["d3_logprob"] = df["nll"]
    df["d4_verbal"] = 1 - df["verbal_p"] / 100
    df["d5_dq"] = 1 - (df["dq1"] + df["dq2"] + df["dq3"]) / 3
    df["d6_iq"] = 1 - df["iq_lm"]
    df["j_nli"], df["j_str"] = jn, js
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["pilot", "e2"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nli-batch", type=int, default=10)
    ap.add_argument("--mailto", default="")
    args = ap.parse_args()
    set_seed(CFG["seeds"]["global"])
    df, recs = load(args.split)
    print(f"{len(recs)} topics, {len(df)} references; parse methods: {df['parse_method'].value_counts().to_dict()}")

    o = CFG["oracle"]
    oracle = Oracle(ROOT / "results/raw/oracle_cache.jsonl", mailto=args.mailto)
    lab = pd.DataFrame([oracle.label(t, float(o["tau"]), float(o["partial_band"])) for t in df["title"]])
    df = pd.concat([df, lab], axis=1)
    fam = {t: oracle.openalex_count(t.split(":", 1)[-1].strip()) for t in df["topic"].unique()}
    df["familiarity_log10"] = np.log10(df["topic"].map(fam) + 1)
    df = score(df, recs, args.device, args.nli_batch)
    df["y"] = (df["status"] == "not_found").astype(int)
    df.drop(columns=["iq"]).to_csv(ROOT / f"results/e2_refs_{args.split}.csv", index=False)

    print(pd.crosstab(df["status"], df["parse_method"], margins=True).to_string())
    missing = {c: int(df[c].isna().sum()) for c in DETECTORS}
    print("missing detector values (counted, excluded per detector):", missing)
    if args.split == "pilot":
        print(df[DETECTORS].describe().round(3).to_string())   # pipeline check only; no AUROC on pilot
        return

    prim = df[df["status"] != "partial"].reset_index(drop=True)
    seed, nb = CFG["seeds"]["bootstrap"], CFG["metrics"]["bootstrap_resamples"]
    # complete cases across all detectors so every comparison is on the same references
    cc = prim.dropna(subset=DETECTORS).reset_index(drop=True)
    res = cluster_bootstrap_auc({c: cc[c] for c in DETECTORS}, cc["y"], cc["topic"], nb, seed)
    rows = [{"detector": c, "n": len(cc), "n_not_found": int(cc["y"].sum()), "auroc": round(res[c]["auc"], 4),
             "ci_lo": round(res[c]["lo"], 4), "ci_hi": round(res[c]["hi"], 4),
             "auc_pr": round(average_precision_score(cc["y"], cc[c]), 4), "prevalence": round(cc["y"].mean(), 4)}
            for c in DETECTORS]
    pd.DataFrame(rows).to_csv(ROOT / "results/e2_auroc.csv", index=False)
    deltas = [{"contrast": f"{a} - {b}", **{k: round(v, 4) for k, v in paired_delta(res, a, b).items()}}
              for a, b in [("d1_nli", "d6_iq"), ("j_nli", "d6_iq"), ("d1_nli", "d5_dq")]]
    for d in deltas:
        d["ci_excludes_0"] = d["lo"] > 0 or d["hi"] < 0
    pd.DataFrame(deltas).to_csv(ROOT / "results/e2_deltas.csv", index=False)

    # RQ2: fabrication vs familiarity (logistic slope, topic-cluster bootstrap) + tertiles
    from sklearn.linear_model import LogisticRegression
    x, yv, top = prim[["familiarity_log10"]].values, prim["y"].values, prim["topic"].values
    slope = lambda xx, yy: LogisticRegression(C=1e6).fit(xx, yy).coef_[0, 0]
    groups = [np.flatnonzero(top == t) for t in np.unique(top)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(nb):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(yv[idx])) == 2:
            boots.append(slope(x[idx], yv[idx]))
    prim["tertile"] = pd.qcut(prim["familiarity_log10"].rank(method="first"), 3, labels=["low", "mid", "high"])
    fam_rows = [{"slope_log10": round(slope(x, yv), 4), "ci_lo": round(np.percentile(boots, 2.5), 4),
                 "ci_hi": round(np.percentile(boots, 97.5), 4), "n": len(prim)}]
    tert = prim.groupby("tertile", observed=True).agg(n=("y", "size"), fabrication_rate=("y", "mean")).reset_index()
    pd.concat([pd.DataFrame(fam_rows), tert], axis=1).to_csv(ROOT / "results/e2_familiarity.csv", index=False)

    # RQ3: D1 vs number of samples N (fixed subsets, seed 11)
    srng = np.random.default_rng(CFG["seeds"]["n_subsample"])
    n_rows = []
    for n in CFG["metrics"]["n_sensitivity"]:
        sub = srng.choice(CFG["generator"]["n_samples"], n, replace=False)
        s_n = [float(np.mean(np.array(v)[sub])) for v in cc["nli_per_sample"]]
        r_n = cluster_bootstrap_auc({"d1": s_n}, cc["y"], cc["topic"], nb, seed)["d1"]
        n_rows.append({"n_samples": n, "auroc": round(r_n["auc"], 4), "ci_lo": round(r_n["lo"], 4), "ci_hi": round(r_n["hi"], 4)})
    pd.DataFrame(n_rows).to_csv(ROOT / "results/e2_n_sensitivity.csv", index=False)

    # cost: generation seconds per reference by component (scoring time not included)
    cost = pd.DataFrame([t["timing_sec"] | {"n_titles": t["n_titles"]} for t in recs])
    per_ref = (cost.drop(columns="n_titles").sum() / cost["n_titles"].sum()).round(3)
    per_ref.rename("gen_seconds_per_reference").to_csv(ROOT / "results/e2_cost.csv")

    with pd.option_context("display.width", 200):
        print(pd.DataFrame(rows).to_string(index=False), "\n")
        print(pd.DataFrame(deltas).to_string(index=False), "\n")
        print(pd.read_csv(ROOT / "results/e2_familiarity.csv").to_string(index=False), "\n")
        print(pd.DataFrame(n_rows).to_string(index=False), "\n")
        print(per_ref.to_string())


if __name__ == "__main__":
    main()
