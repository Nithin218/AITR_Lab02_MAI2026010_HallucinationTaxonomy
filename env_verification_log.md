# Environment Verification Log (Prompt 08)

Paste real console output only. One entry per run.

## What the environment test checks and its expected output

`python scripts/00_env_check.py --mailto <email> --qwen` prints one `[PASS]/[FAIL]/[SKIP] <check>` line per check, each followed by a JSON detail line. It ends with a SUMMARY and writes `results/env/env_check_<UTC>.json`.

| Check | Expected output type on success | Common failures and what to do |
|---|---|---|
| `system` | dict: python, platform, nvidia_smi string, `torch_cuda_available: true`, gpu name (e.g. Tesla T4), gpu_mem_gb ≈ 15 | `torch_cuda_available: false`: Runtime → Change runtime type → T4 GPU |
| `packages` | dict {package: version string}; accelerate/bitsandbytes may be null on CPU | FAIL lists missing packages: rerun the pip cell |
| `pip_check` | the string "No broken requirements found." | Conflicts, often selfcheckgpt pulling `openai`/`groq`/`spacy` versions against Colab's. Note the pair, then pin the lower one in requirements.txt and log it as a deviation. |
| `seed_control` | dict of 3 identical seeded draws (numpy, torch cpu, torch cuda) | FAIL = draws differ (should not happen); report it |
| `agrawal_data` | per-file dict: rows = 1000 (`.titles`: lines = 200), `bing_return_False` count, columns, sha256 | HTTP error: GitHub rate limit or wrong commit; row-count mismatch: file corrupted, delete `data/raw/agrawal` and rerun |
| `wikibio_data` | rows 238, samples_per_passage [20], label_counts over 3 labels | HF hub unreachable: retry; `datasets` version too new or old: note the version |
| `nli_label_order` | p_contradiction for the entailed pair is low (< 0.5) and for the contradictory pair is high | FAIL means our assumption (index 1 = contradiction) is wrong. **Stop**, because it affects D1, J-NLI and V1. |
| `selfcheck_smoke` | lists of per-sentence scores; scores in [0, 1]; seconds per NLI pair; unigram output keys `sent_level`/`doc_level` | `OSError: [E050] Can't find model 'en_core_web_sm'`: run the spaCy download cell; CUDA OOM: restart the runtime |
| `oracle_openalex_crossref` | 10 rows with best (similarity, title) per source. Expect real titles to score high and invented ones low. **This is descriptive, not pass/fail.** | HTTP 429: rate limited, rerun later; SKIP: pass `--mailto` |
| `qwen_4bit_timing` | revision hash; load seconds; greedy tokens/sec; batch-of-10 seconds; peak VRAM GB (expect < 15); first 400 characters of output | `ImportError bitsandbytes`: reinstall and restart the runtime; CUDA OOM: restart, or test the 1.5B fallback; very slow (< 5 tok/s): note it, since it changes the compute budget |

---

## Runs

### Run 0: local Mac, partial (AI assistant, 2026-10-29)
These checks ran locally only and are not a substitute for the Colab run. The individual check functions were called directly, so no JSON file was written.

- Machine: macOS 26.7.1 arm64, Python 3.13.5, no CUDA
- `py_compile` on all scripts and `src/`: OK. Notebook JSON parses.
- `scripts/01_make_pilot_split.py`: `pilot=10 main=190 sha256=94422a812ca0`. Pilot topics are in `configs/pilot_topics.txt`. **The Colab run must print the same 10 topics.**
- `[PASS] system`, `[PASS] seed_control`, `[PASS] pip_check` ("No broken requirements found.", local env only)
- `[PASS] agrawal_data` at commit `bf3ef5f`. Every file has 1,000 rows (`.titles`: 200). `bing_return == "False"` counts:

  | File | False / 1000 | Paper's reported hallucination rate |
  |---|---|---|
  | gpt-4 | 468 | 46.8% |
  | gpt-3.5-turbo | 596 | 59.6% |
  | text-davinci-003 | 736 | 73.6% |
  | llama-2-7b-chat | 683 | 68.3% |
  | llama-2-13b-chat | 767 | 76.7% |
  | llama-2-70b-chat | 662 | 66.2% |

  All six match the paper (data integrity only).
- **New pitfall found:** the three Llama files do not share one schema. 13B has `DQ1_ans` where 7B has `IQ_full_ans1`, and 70B lacks `bing_return` among its first columns. The loader in Prompt 09 must map columns per file explicitly.
- Not run locally: packages (`datasets`, `selfcheckgpt`, `rapidfuzz` not installed), wikibio, NLI label order, selfcheck smoke, oracle (needs your `--mailto`), Qwen (needs a GPU).

### Run 1: Colab T4 (student, 2026-10-29 19:18 UTC)
JSON: `results/env/env_check_20261029T191853Z.json` (on Drive; copy it into the repo).

| Check | Result | Key output | Interpretation |
|---|---|---|---|
| pilot split | same as Run 0 | `pilot=10 main=190 sha256=94422a812ca0`; identical 10 topics | **VERIFIED** deterministic across machines |
| system | PASS | Python 3.13.16; Tesla T4 15.6 GB; driver 580.82.07; torch CUDA 13.0 | — |
| packages | PASS | torch 2.11.0+cu130, transformers 4.57.6, datasets 4.8.5, selfcheckgpt 0.1.7, spacy 3.8.16, bert_score 0.3.13, nltk 3.9.1, sentencepiece 0.2.2, rapidfuzz 3.14.6, requests 2.32.4, pandas 2.2.3, numpy 2.1.3, scikit-learn 1.6.1, pyyaml 6.0.3, accelerate 1.15.0, bitsandbytes 0.50.2 | Exact versions; full list in `requirements-lock.txt` (copy from Drive) |
| pip_check | **FAIL** | ipython→jedi missing; gradio, gradio-client and diffusers need huggingface-hub ≥ 1.16/1.32, installed 0.36.2 | All four are preinstalled Colab packages we never import. Debugged in D-012; check fixed. |
| seed_control | PASS | numpy/torch-cpu draws **identical to the Mac run**; CUDA draws reproducible within the run | — |
| agrawal_data | PASS | Checksums identical to Run 0 | — |
| wikibio_data | PASS | 238 rows; 1,908 sentences; major 761 / minor 631 / accurate 516; 20 samples each | (761+631)/1908 = **72.96%** = README "random" NonFact AUC-PR, and 516/1908 = 27.04% = random Factual. So **NonFact = minor + major inaccurate (VERIFIED)**. |
| nli_label_order | PASS | P(contra): entailed pair 0.0002, contradictory pair 0.9997; `id2label` is generic LABEL_0/1 | **VERIFIED** index 1 = contradiction |
| selfcheck_smoke | PASS | 5 passages × 20 samples; scores in [0.222, 0.998]; **0.0761 s per NLI pair**; unigram keys `sent_level{avg,max}_neg_logprob` | Full V1 ≈ 38,160 pairs ≈ **48 min** |
| oracle | PASS | All 5 real titles: OpenAlex similarity 100. Crossref missed "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (best 80, a different paper). **Invented-title rows were cut off in the paste.** | OpenAlex-first ordering is justified. Need the invented rows from the JSON. |
| qwen_4bit_timing | PASS | revision `a09a35458c702b33eeacc393d103063234e8bc28`; load 235 s (mostly download); greedy **15.1 tok/s**; batch of 10 × 400 tokens in 50.1 s; **peak VRAM 8.09 GB** | Fits on a T4. The greedy answer **hit the 400-token cap** and is in "Author (year). **Title**. Venue" markdown with descriptions (see D-013). |

Harmless noise in the log: a TensorFlow message "Could not find cuda drivers… GPU will not be used" comes from TensorFlow being imported indirectly. PyTorch used the GPU ("SelfCheck-NLI initialized to device cuda").
