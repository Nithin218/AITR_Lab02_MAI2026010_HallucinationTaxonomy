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

### Run 1: Colab (student)
_Paste the full output of notebook cells 4 and 5 here._
