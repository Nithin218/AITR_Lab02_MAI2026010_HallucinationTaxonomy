"""E2 step 1: generate everything the detectors need with the open-weight generator (protocol §4, PC-003).

Usage (repo root):
    python scripts/14_e2_generate.py --split pilot --limit 2     # parse/timing check on pilot topics
    python scripts/14_e2_generate.py --split e2                   # the 100 E2 topics
One JSON per topic in results/raw/e2_<split>/ (a rerun skips finished topics). No scoring happens here.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr import prompts, splits  # noqa: E402
from aitr.seed import best_device, set_seed  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/protocol_v1.yaml").read_text())
G = CFG["generator"]


class Generator:
    def __init__(self, name: str, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        self.device = device
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def _enc(self, dialogs):
        texts = [self.tok.apply_chat_template(d, add_generation_prompt=True, tokenize=False) for d in dialogs]
        return self.tok(texts, return_tensors="pt", padding=True).to(self.device)

    @torch.no_grad()
    def generate(self, dialogs, n=1, sample=False, max_new_tokens=64, temperature=1.0, top_p=1.0):
        enc = self._enc(dialogs)
        kw = dict(do_sample=True, temperature=temperature, top_p=top_p) if sample else \
            dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, num_return_sequences=n,
                                  pad_token_id=self.tok.eos_token_id, **kw)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def span_nll(self, dialog, answer: str, spans: list[tuple[int, int]]) -> list:
        """Teacher-forced mean token NLL of each character span of `answer` given the prompt (detector D3)."""
        prompt = self.tok.apply_chat_template(dialog, add_generation_prompt=True, tokenize=False)
        p_ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        a = self.tok(answer, return_offsets_mapping=True, add_special_tokens=False)
        ids = torch.cat([p_ids, torch.tensor(a["input_ids"])]).unsqueeze(0).to(self.device)
        logp = torch.log_softmax(self.model(ids).logits[0, :-1].float(), dim=-1)
        tok_nll = -logp[torch.arange(ids.shape[1] - 1), ids[0, 1:]][len(p_ids) - 1:].cpu()
        out = []
        for s, e in spans:
            sel = [i for i, (a0, a1) in enumerate(a["offset_mapping"]) if a1 > s and a0 < e]
            out.append(float(tok_nll[sel].mean()) if sel else None)
        return out


def run_topic(gen: Generator, topic: str, seed: int) -> dict:
    timing = {}
    set_seed(seed)
    t = time.time()
    q = prompts.prompt_Q(topic)
    main = gen.generate([q], max_new_tokens=G["main_max_new_tokens"])[0]
    titles, how = prompts.extract_titles(main, G["k_titles"])
    timing["main"] = time.time() - t

    t = time.time()
    samples = gen.generate([q], n=G["n_samples"], sample=True, max_new_tokens=G["main_max_new_tokens"],
                           temperature=G["temperature"], top_p=G["top_p"])
    timing["samples"] = time.time() - t

    spans = [(main.find(ti), main.find(ti) + len(ti)) if ti in main else (-1, -1) for ti in titles]
    t = time.time()
    nll = gen.span_nll(q, main, [s for s in spans if s[0] >= 0])
    nll_iter = iter(nll)
    nll = [next(nll_iter) if s[0] >= 0 else None for s in spans]
    timing["logprob"] = time.time() - t

    refs = []
    for pos, title in enumerate(titles, start=1):
        r = {"pos": pos, "title": title, "nll": nll[pos - 1]}
        t = time.time()
        for name, dialog in (("dq1", prompts.prompt_DQ1_sample(title)), ("dq2", prompts.prompt_DQ2_sample(title)),
                             ("dq3", prompts.prompt_DQ3_sample(main, pos))):
            r[name] = gen.generate([dialog], n=G["dq_samples"], sample=True, max_new_tokens=8,
                                   temperature=G["temperature"], top_p=G["agrawal_top_p"])
        timing["dq"] = timing.get("dq", 0) + time.time() - t
        t = time.time()
        r["iq"] = gen.generate([prompts.prompt_IQ(title)], n=G["iq_samples"], sample=True, max_new_tokens=100,
                               temperature=G["temperature"], top_p=G["agrawal_top_p"])
        pairs = [(i, j) for i in range(len(r["iq"])) for j in range(i)]   # as consistency_check()
        r["iq_judge"] = gen.generate([prompts.prompt_consistency(r["iq"][i], r["iq"][j]) for i, j in pairs],
                                     max_new_tokens=150)
        r["iq_frac"] = [prompts.get_agreement_frac(a) for a in r["iq_judge"]]
        timing["iq"] = timing.get("iq", 0) + time.time() - t
        t = time.time()
        r["verbal"] = gen.generate([prompts.VERBAL_P(title)], max_new_tokens=8)[0]
        r["verbal_p"] = prompts.parse_verbal(r["verbal"])
        timing["verbal"] = timing.get("verbal", 0) + time.time() - t
        refs.append(r)
    return {"topic": topic, "seed": seed, "main_answer": main, "parse_method": how, "n_titles": len(titles),
            "samples": samples, "refs": refs, "timing_sec": {k: round(v, 2) for k, v in timing.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["pilot", "e2"], required=True)
    ap.add_argument("--limit", type=int, default=10**9)
    args = ap.parse_args()
    out_dir = ROOT / f"results/raw/e2_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    topics = splits.topics(args.split)[:args.limit]
    all_topics = splits.topics("pilot") + splits.topics("main")
    device = best_device()
    gen = Generator(G["model"], device)
    print(f"model {G['model']} revision {gen.revision} device {device}", flush=True)
    if G["revision"] not in (None, "TBD") and gen.revision and gen.revision != G["revision"]:
        raise SystemExit(f"revision mismatch: config {G['revision']} vs loaded {gen.revision}")
    for k, topic in enumerate(topics):
        idx = all_topics.index(topic)
        path = out_dir / f"{idx:03d}.json"
        if path.exists():
            continue
        t = time.time()
        rec = run_topic(gen, topic, CFG["seeds"]["global"] + idx)
        rec |= {"topic_index": idx, "model": G["model"], "revision": gen.revision, "device": device}
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{k + 1}/{len(topics)}] {topic}: {rec['n_titles']} titles ({rec['parse_method']}), "
              f"{time.time() - t:.0f}s {rec['timing_sec']}", flush=True)


if __name__ == "__main__":
    main()
