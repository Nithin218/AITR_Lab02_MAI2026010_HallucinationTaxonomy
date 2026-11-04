"""R1: re-label Agrawal et al.'s generated titles with OpenAlex/Crossref and compare with their Bing labels.

Usage (repo root):
    python scripts/13_r1_oracle_relabel.py --split pilot [--tau 90] [--mailto you@uni.edu]
    python scripts/13_r1_oracle_relabel.py --split main  --model llama-2-7b-chat
Responses are cached in results/raw/oracle_cache.jsonl (reruns do not re-query).
Outputs: results/r1_labels_<model>_<split>.csv and, for main, results/r1_agreement.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr import agrawal, splits  # noqa: E402
from aitr.oracle import Oracle  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/protocol_v1.yaml").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["pilot", "main"], required=True)
    ap.add_argument("--model", default="llama-2-7b-chat")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--mailto", default="")
    args = ap.parse_args()
    tau = args.tau if args.tau is not None else float(CFG["oracle"]["tau"])
    band = float(CFG["oracle"]["partial_band"])

    df = agrawal.load_model(ROOT / "data/raw/agrawal", args.model)
    if args.model == "llama-2-13b-chat":
        df = agrawal.drop_duplicate_refs(df)
    df = df[df["topic"].isin(set(splits.topics(args.split)))].reset_index(drop=True)
    (ROOT / "results/raw").mkdir(parents=True, exist_ok=True)
    oracle = Oracle(ROOT / "results/raw/oracle_cache.jsonl", mailto=args.mailto)
    labels = []
    for k, t in enumerate(df["gen_title"]):
        labels.append(oracle.label(t, tau, band))
        if k % 50 == 0:
            print(f"  {k + 1}/{len(df)}", flush=True)
    out = pd.concat([df[["model", "row", "topic", "gen_title", "y"]], pd.DataFrame(labels)], axis=1)
    out["bing"] = out["y"].map({0: "grounded", 1: "hallucinated"})
    out.to_csv(ROOT / f"results/r1_labels_{args.model}_{args.split}.csv", index=False)

    print(f"\ntau={tau}, partial band=[{tau - band}, {tau})")
    print(pd.crosstab(out["bing"], out["status"], margins=True).to_string())
    binary = out[out["status"] != "partial"]
    kappa = cohen_kappa_score(binary["y"], (binary["status"] == "not_found").astype(int))
    print(f"Cohen's kappa (Bing vs oracle, partial excluded, n={len(binary)}): {kappa:.3f}")
    if args.split == "pilot":
        # For setting tau: inspect the disagreements and the near-threshold cases by hand.
        show = out[(out["status"] == "partial") | ((out["status"] == "found") != (out["y"] == 0))]
        with pd.option_context("display.max_colwidth", 70, "display.width", 250):
            print(show[["bing", "status", "best_sim", "gen_title", "best_title"]].to_string(index=False))
    else:
        pd.DataFrame([{"model": args.model, "tau": tau, "n": len(out), "n_partial": int((out["status"] == "partial").sum()),
                       "kappa_excl_partial": round(kappa, 4)}]).to_csv(ROOT / "results/r1_agreement.csv", index=False)


if __name__ == "__main__":
    main()
