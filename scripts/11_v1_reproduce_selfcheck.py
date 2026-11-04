"""V1: reproduce SelfCheck-NLI and SelfCheck-Unigram on wiki_bio_gpt3_hallucination (GPU recommended).

Usage (repo root):  python scripts/11_v1_reproduce_selfcheck.py [--limit N]
Per-passage scores are checkpointed to results/raw/v1_selfcheck_scores.jsonl, so a rerun after a
Colab disconnect resumes where it stopped. Metrics: results/v1_selfcheck.csv.
Targets and tolerance: baseline_validation.md (pre-registered before the first run).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr.seed import best_device, set_seed  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/protocol_v1.yaml").read_text())
TOLERANCE = 2.0  # AUC-PR points (research brief, validation gate)
# selfcheckgpt README (fetched 2026-10-29): NonFact AUC-PR, Factual AUC-PR, passage-level PCC
TARGETS = {"SelfCheck-NLI": (92.50, 66.08, 74.14), "SelfCheck-Unigram (max)": (85.63, 58.47, 64.71)}
HUMAN = {"accurate": 0.0, "minor_inaccurate": 0.5, "major_inaccurate": 1.0}  # passage score mapping (INFERRED)


class BatchedNLI:
    """Same model, pair order (sentence, sample) and P(contradiction) as SelfCheckNLI, scored in batches.

    SelfCheckNLI encodes one pair per forward pass, which is ~0.5 s/pair on an Apple GPU (5-7 h for V1).
    Equivalence with the official class is checked by --check-equivalence (baseline_validation.md).
    """

    def __init__(self, device, batch_size=8):
        # batch size only affects speed/memory (whole passages as premises; 8 fits an 8 GB Apple GPU)
        from aitr.judges import DebertaContradiction
        b = CFG["baseline"]
        self.scorer = DebertaContradiction(b["nli_model"], device, b["nli_max_length"], batch_size)

    def predict(self, sentences, sampled_passages):
        prem = [p for _ in sentences for p in sampled_passages]
        hyp = [s for s in sentences for _ in sampled_passages]
        return self.scorer(prem, hyp).reshape(len(sentences), len(sampled_passages)).mean(axis=1)


def score_passages(ds, limit, out_path, impl="batched", batch_size=8):
    import torch
    from selfcheckgpt.modeling_selfcheck import SelfCheckNgram, SelfCheckNLI
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["idx"]] = rec
    todo = [i for i in range(min(limit, len(ds))) if i not in done]
    print(f"passages: {len(done)} cached, {len(todo)} to score")
    if todo:
        device = torch.device(best_device())
        print(f"device: {device}")
        nli = (SelfCheckNLI(device=device) if impl == "official" else
               None if impl == "none" else BatchedNLI(str(device), batch_size))
        uni = SelfCheckNgram(n=1)
        t0 = time.time()
        with open(out_path, "a", encoding="utf-8") as f:
            for k, i in enumerate(todo):
                row = ds[i]
                nli_s = [] if nli is None else nli.predict(sentences=row["gpt3_sentences"],
                                                            sampled_passages=row["gpt3_text_samples"])
                uni_s = uni.predict(sentences=row["gpt3_sentences"], passage=row["gpt3_text"],
                                    sampled_passages=row["gpt3_text_samples"])
                rec = {"idx": i, "nli": [float(x) for x in nli_s] if nli is not None else None,
                       "uni_max": [float(x) for x in uni_s["sent_level"]["max_neg_logprob"]],
                       "uni_avg": [float(x) for x in uni_s["sent_level"]["avg_neg_logprob"]]}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[i] = rec
                if k % 20 == 0:
                    print(f"  {k + 1}/{len(todo)} passages, {time.time() - t0:.0f}s", flush=True)
    return [done[i] for i in sorted(done) if i < limit]


def evaluate(ds, recs):
    ann = [a for r in recs for a in ds[r["idx"]]["annotation"]]
    nonfact = np.array([a != "accurate" for a in ann], dtype=int)
    human = [np.mean([HUMAN[a] for a in ds[r["idx"]]["annotation"]]) for r in recs]
    rows = []
    for method, key in (("SelfCheck-NLI", "nli"), ("SelfCheck-Unigram (max)", "uni_max"),
                        ("SelfCheck-Unigram (avg)", "uni_avg")):
        if recs[0][key] is None:
            continue
        s = np.array([x for r in recs for x in r[key]])
        assert len(s) == len(nonfact), "sentence/score misalignment"
        passage = [np.mean(r[key]) for r in recs]
        ours = (100 * average_precision_score(nonfact, s), 100 * average_precision_score(1 - nonfact, -s),
                100 * pearsonr(passage, human)[0])
        target = TARGETS.get(method)
        row = {"method": method, "n_passages": len(recs), "n_sentences": len(s),
               "nonfact_aucpr": round(ours[0], 2), "factual_aucpr": round(ours[1], 2), "pcc": round(ours[2], 2)}
        if target:
            row |= {"target_nonfact": target[0], "target_factual": target[1], "target_pcc": target[2],
                    "nonfact_diff": round(ours[0] - target[0], 2), "factual_diff": round(ours[1] - target[1], 2),
                    "pass_nonfact": abs(ours[0] - target[0]) <= TOLERANCE}
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10**9, help="score only the first N passages (smoke test)")
    ap.add_argument("--impl", choices=["batched", "official", "none"], default="batched",
                    help="NLI implementation: batched (default) or the official one-pair-at-a-time SelfCheckNLI")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--check-equivalence", action="store_true",
                    help="compare batched NLI with the cached official scores and exit")
    args = ap.parse_args()
    set_seed(CFG["seeds"]["global"])
    from datasets import load_dataset
    ds = load_dataset(CFG["data"]["wikibio_dataset"], split=CFG["data"]["wikibio_split"])
    (ROOT / "results/raw").mkdir(parents=True, exist_ok=True)
    official_path = ROOT / "results/raw/v1_selfcheck_scores_official.jsonl"
    if args.check_equivalence:
        official = [json.loads(l) for l in official_path.read_text(encoding="utf-8").splitlines()]
        nli = BatchedNLI(best_device(), args.batch_size)
        diffs = []
        for rec in official:
            row = ds[rec["idx"]]
            ours = nli.predict(row["gpt3_sentences"], row["gpt3_text_samples"])
            diffs += list(np.abs(np.array(ours) - np.array(rec["nli"])))
        print(f"equivalence: {len(official)} passages, {len(diffs)} sentence scores, "
              f"max |batched - official| = {max(diffs):.2e}, mean = {np.mean(diffs):.2e}")
        return
    path = {"official": official_path, "none": ROOT / "results/raw/v1_selfcheck_scores_unigram_only.jsonl"
            }.get(args.impl, ROOT / "results/raw/v1_selfcheck_scores.jsonl")
    recs = score_passages(ds, args.limit, path, args.impl, args.batch_size)
    out = evaluate(ds, recs)
    suffix = ("" if len(recs) == len(ds) else f"_first{len(recs)}") + ("_unigram_only" if args.impl == "none" else "")
    path = ROOT / f"results/v1_selfcheck{suffix}.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"written {path.relative_to(ROOT)}" + ("" if not suffix else "  (PARTIAL: not a V1 result)"))


if __name__ == "__main__":
    main()
