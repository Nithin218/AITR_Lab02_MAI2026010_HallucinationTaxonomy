"""E1: judge-swap ablation on Agrawal et al.'s released author samples (protocol §1, §3).

Same 3 author answers per reference; only the consistency judge changes: J-LM (released) vs J-NLI vs J-STR.

Usage (repo root):
    python scripts/12_e1_judge_swap.py --split pilot --judges str      # pipeline check, no AUROC printed
    python scripts/12_e1_judge_swap.py --split main  --judges str,nli  # the experiment (after protocol-v1)
NLI scores are checkpointed per model to results/raw/e1_nli_<model>_<split>.csv.
Outputs: results/e1_scores_<split>.csv (per reference) and, for main, results/e1_auroc.csv + e1_deltas.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr import agrawal, judges, splits  # noqa: E402
from aitr.metrics import cluster_bootstrap_auc, paired_delta  # noqa: E402
from aitr.seed import set_seed  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/protocol_v1.yaml").read_text())
IQ = ["iq_1", "iq_2", "iq_3"]


def load(split: str) -> pd.DataFrame:
    keep = set(splits.topics(split))
    frames = []
    for model in agrawal.MODELS:
        df = agrawal.load_model(ROOT / "data/raw/agrawal", model)
        if model == "llama-2-13b-chat":
            df = agrawal.drop_duplicate_refs(df)  # D-014
        frames.append(df[df["topic"].isin(keep)])
    return pd.concat(frames, ignore_index=True)


def nli_text(s: str) -> str:
    """NLI input: whitespace collapsed. J-STR uses the raw text, whose newlines separate names."""
    return " ".join(s.split())


def nli_scores(df: pd.DataFrame, model: str, split: str, scorer) -> pd.Series:
    path = ROOT / f"results/raw/e1_nli_{model}_{split}.csv"
    part = df[df["model"] == model]
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["row", "j_nli", "j_nli_raw"])
    have = set(cached["row"])
    todo = part[~part["row"].isin(have)]
    print(f"  {model}: {len(have)} cached, {len(todo)} to score", flush=True)
    new = []
    for k, (_, r) in enumerate(todo.iterrows()):
        rule, raw = judges.j_nli([nli_text(r[c]) for c in IQ], scorer)
        new.append({"row": r["row"], "j_nli": rule, "j_nli_raw": raw})
        if len(new) == 100 or k == len(todo) - 1:
            cached = pd.concat([cached, pd.DataFrame(new)], ignore_index=True)
            cached.to_csv(path, index=False)
            new = []
    cached["row"] = cached["row"].astype(int)
    return part.merge(cached, on="row", how="left")[["j_nli", "j_nli_raw"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["pilot", "main"], required=True)
    ap.add_argument("--judges", default="str,nli")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    set_seed(CFG["seeds"]["global"])
    use = set(args.judges.split(","))
    df = load(args.split)
    df["j_lm"] = 1 - df["iq_lm"]
    if "str" in use:
        df["j_str"] = [judges.j_str([r[c] for c in IQ]) for _, r in df.iterrows()]
    if "nli" in use:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        scorer = judges.DebertaContradiction(CFG["baseline"]["nli_model"], device,
                                             CFG["baseline"]["nli_max_length"], CFG["baseline"]["nli_batch_size"])
        (ROOT / "results/raw").mkdir(parents=True, exist_ok=True)
        df["j_nli"], df["j_nli_raw"] = np.nan, np.nan
        for model in agrawal.MODELS:
            m = df["model"] == model
            df.loc[m, ["j_nli", "j_nli_raw"]] = nli_scores(df, model, args.split, scorer).values
    cols = ["model", "row", "topic", "gen_title", "y"] + [c for c in ("j_lm", "j_str", "j_nli", "j_nli_raw") if c in df]
    df[cols].to_csv(ROOT / f"results/e1_scores_{args.split}.csv", index=False)

    if args.split == "pilot":
        # Pipeline and name-parsing diagnostics only: pilot AUROC is not looked at (protocol §4).
        df["n_tokens"] = [[len(judges.name_tokens(r[c])) for c in IQ] for _, r in df.iterrows()]
        print(df.groupby("model").agg(refs=("y", "size"), hallucinated=("y", "sum")).to_string())
        print("answers parsing to 0 name tokens:", int(sum(n == 0 for ns in df["n_tokens"] for n in ns)),
              "of", 3 * len(df), "| references with all 3 empty:",
              int(sum(all(n == 0 for n in ns) for ns in df["n_tokens"])))
        print(df[[c for c in ("j_lm", "j_str", "j_nli", "j_nli_raw") if c in df]].describe().round(3).to_string())
        return

    judge_cols = [c for c in ("j_lm", "j_nli", "j_str", "j_nli_raw") if c in df]
    auc_rows, delta_rows = [], []
    for model in agrawal.MODELS:
        part = df[df["model"] == model]
        res = cluster_bootstrap_auc({c: part[c] for c in judge_cols}, part["y"], part["topic"],
                                    CFG["metrics"]["bootstrap_resamples"], CFG["seeds"]["bootstrap"])
        for c in judge_cols:
            auc_rows.append({"model": model, "judge": c, "n": len(part), "n_hallucinated": int(part["y"].sum()),
                             "auroc": round(res[c]["auc"], 4), "ci_lo": round(res[c]["lo"], 4),
                             "ci_hi": round(res[c]["hi"], 4)})
        for a in [c for c in judge_cols if c != "j_lm"]:
            d = paired_delta(res, a, "j_lm")
            delta_rows.append({"model": model, "contrast": f"{a} - j_lm", "delta": round(d["delta"], 4),
                               "ci_lo": round(d["lo"], 4), "ci_hi": round(d["hi"], 4),
                               "ci_excludes_0": d["lo"] > 0 or d["hi"] < 0})
    auc, deltas = pd.DataFrame(auc_rows), pd.DataFrame(delta_rows)
    auc.to_csv(ROOT / "results/e1_auroc.csv", index=False)
    deltas.to_csv(ROOT / "results/e1_deltas.csv", index=False)
    print(auc.to_string(index=False), "\n")
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
