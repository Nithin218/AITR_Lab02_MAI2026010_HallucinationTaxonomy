"""POST-HOC (not pre-registered; peer review M2): does label noise explain near-chance E1 AUROC on Llama-2-7B?

Recomputes E1 judge AUROCs for Llama-2-7B on references whose Bing label is confirmed by the
OpenAlex/Crossref oracle (R1): Bing-grounded & oracle-found, or Bing-hallucinated & oracle-not-found.
Usage (repo root):  python scripts/17_posthoc_label_agreement.py
Output: results/posthoc_label_agreement.csv
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr.metrics import cluster_bootstrap_auc, paired_delta  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/protocol_v1.yaml").read_text())
MODEL = "llama-2-7b-chat"
JUDGES = ["j_lm", "j_nli", "j_str"]


def main():
    r1 = pd.read_csv(ROOT / f"results/r1_labels_{MODEL}_main.csv")
    e1 = pd.read_csv(ROOT / "results/e1_scores_main.csv")
    e1 = e1[e1["model"] == MODEL]
    df = e1.merge(r1[["row", "status"]], on="row", how="inner", validate="one_to_one")
    agree = ((df["y"] == 0) & (df["status"] == "found")) | ((df["y"] == 1) & (df["status"] == "not_found"))
    rows = []
    for name, part in (("all main refs (E1)", df), ("Bing and oracle agree", df[agree])):
        part = part.reset_index(drop=True)
        res = cluster_bootstrap_auc({j: part[j] for j in JUDGES}, part["y"], part["topic"],
                                    CFG["metrics"]["bootstrap_resamples"], CFG["seeds"]["bootstrap"])
        row = {"subset": name, "n": len(part), "n_hallucinated": int(part["y"].sum())}
        for j in JUDGES:
            row[f"{j}_auroc"] = round(res[j]["auc"], 4)
            row[f"{j}_ci"] = f"[{res[j]['lo']:.3f}, {res[j]['hi']:.3f}]"
        d = paired_delta(res, "j_nli", "j_lm")
        row["j_nli_minus_j_lm"] = f"{d['delta']:+.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/posthoc_label_agreement.csv", index=False)
    with pd.option_context("display.width", 250):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
