"""Environment check (assignment Prompt 08). Computes no experimental metric.

Usage (repo root):
    python scripts/00_env_check.py --mailto you@example.com          # all checks except Qwen
    python scripts/00_env_check.py --mailto you@example.com --qwen   # also load Qwen2.5-7B 4-bit (GPU)

Each check prints [PASS] / [FAIL] / [SKIP]. The full record is written to
results/env/env_check_<UTC timestamp>.json.
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from aitr.download import fetch_agrawal  # noqa: E402
from aitr.seed import set_seed  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/protocol_v1.yaml").read_text())
REQUIRED = ["torch", "transformers", "datasets", "selfcheckgpt", "spacy", "bert_score", "nltk",
            "sentencepiece", "rapidfuzz", "requests", "pandas", "numpy", "scikit-learn", "pyyaml"]
GPU_ONLY = ["accelerate", "bitsandbytes"]

# Titles for the oracle smoke test only. REAL = entries from the T20 review's references.bib.
# INVENTED = written for this test and believed not to exist. Not part of the pilot or main set.
REAL_TITLES = [
    "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models",
    "Survey of Hallucination in Natural Language Generation",
    "FActScore: Fine-Grained Atomic Evaluation of Factual Precision in Long Form Text Generation",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "On Faithfulness and Factuality in Abstractive Summarization",
]
INVENTED_TITLES = [
    "Lattice-Guided Citation Repair for Federated Bibliographic Graphs",
    "Thermodynamic Bounds on Reference Hallucination in Sparse Transformers",
    "A Quantum Annealing Approach to Peer Review Assignment in Low-Resource Venues",
    "Recursive Provenance Folding for Multilingual Scholarly Knowledge Bases",
    "Adversarial Bibliometrics: Detecting Synthetic Authorship with Topological Signatures",
]
QWEN_TEST_TOPIC = "Information systems: Data cleaning"  # must NOT be one of the 200 study topics


class Skip(Exception):
    pass


RESULTS = []


def run(name, fn, *args):
    t0 = time.time()
    try:
        detail, status = fn(*args), "PASS"
    except Skip as e:
        detail, status = str(e), "SKIP"
    except Exception as e:  # report every failure, keep going
        traceback.print_exc()
        detail, status = f"{type(e).__name__}: {e}", "FAIL"
    RESULTS.append({"check": name, "status": status, "seconds": round(time.time() - t0, 1), "detail": detail})
    print(f"[{status}] {name} ({RESULTS[-1]['seconds']}s)")
    print("   ", json.dumps(detail, ensure_ascii=False)[:1500])
    return detail if status == "PASS" else None


def check_system():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = smi.stdout.strip() or smi.stderr.strip()
    except FileNotFoundError:
        info["nvidia_smi"] = "not found (no NVIDIA GPU)"
    import torch
    info["torch_cuda_available"] = torch.cuda.is_available()
    info["torch_cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_mem_gb"] = round(p.total_memory / 1e9, 1)
    return info


def check_packages():
    versions, missing = {}, []
    for pkg in REQUIRED + GPU_ONLY:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
            if pkg in REQUIRED:
                missing.append(pkg)
    if missing:
        raise RuntimeError(f"missing required packages {missing}; versions={versions}")
    return versions


def check_pip():
    out = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip()
    if out.returncode != 0:
        raise RuntimeError(f"pip check reports conflicts: {text}")
    return text


def check_seed():
    import numpy as np
    import torch
    draws = []
    for _ in range(2):
        set_seed(CFG["seeds"]["global"])
        draws.append((np.random.rand(3).tolist(), torch.rand(3).tolist(),
                      torch.rand(3, device="cuda").tolist() if torch.cuda.is_available() else None))
    if draws[0] != draws[1]:
        raise RuntimeError(f"seeded draws differ: {draws}")
    return {"numpy": draws[0][0], "torch_cpu": draws[0][1], "torch_cuda": draws[0][2]}


def check_agrawal():
    d = CFG["data"]
    raw = ROOT / "data/raw/agrawal"
    sums = fetch_agrawal(raw, d["agrawal_commit"], d["agrawal_files"])
    csv.field_size_limit(sys.maxsize)
    out = {}
    for name in d["agrawal_files"]:
        path = raw / name
        if name.endswith(".titles"):
            n = len([t for t in path.read_text(encoding="utf-8").splitlines() if t.strip()])
            expected = 200
            out[name] = {"lines": n}
        else:
            with open(path, encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            n, expected = len(rows), 1000
            out[name] = {"rows": n, "bing_return_False": sum(r["bing_return"] == "False" for r in rows),
                         "columns": list(rows[0].keys())}
        out[name]["sha256"] = sums[name]
        if n != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {n}")
    return out


def check_wikibio():
    from collections import Counter

    from datasets import load_dataset
    ds = load_dataset(CFG["data"]["wikibio_dataset"], split=CFG["data"]["wikibio_split"])
    labels = Counter(a for row in ds["annotation"] for a in row)
    info = {"rows": len(ds), "columns": ds.column_names, "n_sentences": sum(labels.values()),
            "label_counts": dict(labels), "samples_per_passage": sorted({len(s) for s in ds["gpt3_text_samples"]})}
    if len(ds) != 238 or info["samples_per_passage"] != [20]:
        raise RuntimeError(f"unexpected dataset shape: {info}")
    return info


def check_nli_sanity():
    """Hand-made pairs to confirm the label order assumed by SelfCheckNLI (index 1 = contradiction)."""
    import torch
    from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
    name = CFG["baseline"]["nli_model"]
    tok = DebertaV2Tokenizer.from_pretrained(name)
    model = DebertaV2ForSequenceClassification.from_pretrained(name).eval()
    premise = "The cat is sleeping on the sofa."
    # SelfCheckNLI encodes (sentence, sample) = (hypothesis, premise); keep the same order.
    pairs = [("A cat is on the sofa.", premise), ("The sofa is empty.", premise)]
    with torch.no_grad():
        enc = tok.batch_encode_plus(pairs, return_tensors="pt", padding=True)
        p_contra = torch.softmax(model(**enc).logits, dim=-1)[:, 1].tolist()
    info = {"model": name, "id2label": model.config.id2label,
            "p_contradiction_entailed_pair": round(p_contra[0], 4),
            "p_contradiction_contradictory_pair": round(p_contra[1], 4)}
    if not p_contra[1] > p_contra[0]:
        raise RuntimeError(f"label order assumption looks wrong: {info}")
    return info


def check_selfcheck_smoke():
    """Run the official scorers on a few passages; reports output shapes only (no AUC)."""
    import torch
    from datasets import load_dataset
    from selfcheckgpt.modeling_selfcheck import SelfCheckNgram, SelfCheckNLI
    gpu = torch.cuda.is_available()
    n_pass, n_samp = (5, 20) if gpu else (1, 5)
    ds = load_dataset(CFG["data"]["wikibio_dataset"], split=CFG["data"]["wikibio_split"]).select(range(n_pass))
    nli = SelfCheckNLI(device=torch.device("cuda" if gpu else "cpu"))
    uni = SelfCheckNgram(n=1)
    t0 = time.time()
    nli_scores = [nli.predict(sentences=r["gpt3_sentences"], sampled_passages=r["gpt3_text_samples"][:n_samp])
                  for r in ds]
    nli_sec = time.time() - t0
    uni_out = uni.predict(sentences=ds[0]["gpt3_sentences"], passage=ds[0]["gpt3_text"],
                          sampled_passages=ds[0]["gpt3_text_samples"][:n_samp])
    n_pairs = sum(len(r["gpt3_sentences"]) for r in ds) * n_samp
    return {"passages": n_pass, "samples": n_samp, "nli_scores_per_passage": [len(s) for s in nli_scores],
            "nli_score_range": [round(float(min(map(min, nli_scores))), 4), round(float(max(map(max, nli_scores))), 4)],
            "nli_seconds_per_pair": round(nli_sec / n_pairs, 4),
            "unigram_keys": {k: list(v.keys()) for k, v in uni_out.items()}}


def check_oracle(mailto):
    import requests
    from rapidfuzz import fuzz
    if not mailto:
        raise Skip("pass --mailto <email> (Crossref polite pool asks for it)")
    headers = {"User-Agent": f"AITR-Lab02-env-check (mailto:{mailto})"}

    def norm(s):
        return "".join(c.lower() if c.isalnum() else " " for c in s).strip()

    rows = []
    for kind, titles in (("real", REAL_TITLES), ("invented", INVENTED_TITLES)):
        for t in titles:
            oa = requests.get("https://api.openalex.org/works", headers=headers, timeout=30,
                              params={"search": t, "per_page": 5, "mailto": mailto})
            oa.raise_for_status()
            oa_titles = [w.get("display_name") or "" for w in oa.json()["results"]]
            cr = requests.get("https://api.crossref.org/works", headers=headers, timeout=30,
                              params={"query.bibliographic": t, "rows": 5, "mailto": mailto})
            cr.raise_for_status()
            cr_titles = [(w.get("title") or [""])[0] for w in cr.json()["message"]["items"]]
            best = lambda cands: max([(fuzz.token_sort_ratio(norm(t), norm(c)), c) for c in cands] or [(0, "")])
            rows.append({"kind": kind, "title": t, "openalex_best": best(oa_titles), "crossref_best": best(cr_titles)})
            time.sleep(0.5)
    return rows


def check_qwen():
    import torch
    if not torch.cuda.is_available():
        raise Skip("no CUDA GPU")
    topics = (ROOT / "data/raw/agrawal/acm_ccs_200.titles").read_text(encoding="utf-8").splitlines()
    assert QWEN_TEST_TOPIC not in topics, "test topic overlaps the study topics"
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    g = CFG["generator"]
    tok = AutoTokenizer.from_pretrained(g["model"])
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        g["model"], device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16))
    load_sec = time.time() - t0
    # Timing prompt only. The study uses Agrawal et al.'s prompts verbatim (protocol §4).
    msgs = [{"role": "user", "content": f'List 5 existing references (paper titles) related to "{QWEN_TEST_TOPIC}".'}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    set_seed(CFG["seeds"]["global"])
    torch.cuda.synchronize(); t0 = time.time()
    greedy = model.generate(ids, max_new_tokens=g["main_max_new_tokens"], do_sample=False)
    torch.cuda.synchronize(); greedy_sec = time.time() - t0
    n_greedy = greedy.shape[1] - ids.shape[1]
    t0 = time.time()
    sampled = model.generate(ids, max_new_tokens=g["main_max_new_tokens"], do_sample=True,
                             temperature=g["temperature"], top_p=g["top_p"], num_return_sequences=g["n_samples"])
    torch.cuda.synchronize(); sample_sec = time.time() - t0
    return {"revision": getattr(model.config, "_commit_hash", None), "load_seconds": round(load_sec, 1),
            "greedy_new_tokens": n_greedy, "greedy_tokens_per_sec": round(n_greedy / greedy_sec, 1),
            "batch10_seconds": round(sample_sec, 1),
            "batch10_new_tokens": int((sampled.shape[1] - ids.shape[1]) * g["n_samples"]),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "greedy_text_head": tok.decode(greedy[0, ids.shape[1]:], skip_special_tokens=True)[:400]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mailto", default="")
    ap.add_argument("--qwen", action="store_true", help="also load Qwen2.5-7B in 4-bit (GPU, ~5 GB download)")
    args = ap.parse_args()

    run("system", check_system)
    run("packages", check_packages)
    run("pip_check", check_pip)
    run("seed_control", check_seed)
    run("agrawal_data", check_agrawal)
    run("wikibio_data", check_wikibio)
    run("nli_label_order", check_nli_sanity)
    run("selfcheck_smoke", check_selfcheck_smoke)
    run("oracle_openalex_crossref", check_oracle, args.mailto)
    if args.qwen:
        run("qwen_4bit_timing", check_qwen)
    else:
        RESULTS.append({"check": "qwen_4bit_timing", "status": "SKIP", "seconds": 0, "detail": "run with --qwen"})

    out_dir = ROOT / "results/env"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"env_check_{stamp}.json"
    path.write_text(json.dumps({"utc": stamp, "args": vars(args), "results": RESULTS}, indent=2, ensure_ascii=False))
    print("\nSUMMARY")
    for r in RESULTS:
        print(f"  {r['status']:4}  {r['check']}")
    print(f"written: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
