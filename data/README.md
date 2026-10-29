# Data

Nothing in `data/raw/` is committed. Scripts download it.

| Dataset | Source | Licence | How it is obtained |
|---|---|---|---|
| Agrawal et al. released results (6 CSVs + `acm_ccs_200.titles`) | github.com/microsoft/hallucinated-references, `data/` at commit `bf3ef5f3de89e49a8a4d31bb3c97ba8c777d9c08` | MIT (repo) | `src/aitr/download.py` → `data/raw/agrawal/`; SHA-256 checksums recorded in `results/env/env_check_*.json` |
| WikiBio GPT-3 hallucination | Hugging Face `potsawee/wiki_bio_gpt3_hallucination`, split `evaluation` | CC-BY-SA-3.0 | `datasets.load_dataset` (HF cache) |
| New generations (E2) | Produced by `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0) | — | `results/raw/*.jsonl`, written by the generation script (later) |
| Oracle lookups | OpenAlex API, Crossref REST API | OpenAlex data CC0 (INFERRED, check); Crossref metadata terms (check) | Cached to `results/raw/oracle_*.jsonl` |

Label convention (Agrawal files): `bing_return == "False"` means **hallucinated** (y = 1).
