# Is the Judge the Bottleneck? Consistency-Based Detection of Fabricated References

AITR Lab 02: implementation study following the T20 review *A Taxonomy of Hallucination Types in Large Language Models for Research Applications*.
Nagireddy Nithin Kumar Reddy (MAI2026010), IIIT Allahabad.

Repository: https://github.com/Nithin218/AITR_Lab02_MAI2026010_HallucinationTaxonomy (public). The protocol freeze is the tag `protocol-v1`.

**Question.** Agrawal et al. (2023) found that LM-judged author-consistency checks detect hallucinated references for GPT-4 but not for 7B–13B Llama-2 models. Keeping the sampled author answers fixed, does an external NLI judge (as in SelfCheckGPT) or a model-free name-overlap judge recover detection? Is fabrication more frequent on less-published topics?

**Main result.** No. On Llama-2-7B every judge stays near chance (AUROC 0.51–0.56). On larger models the NLI judge is significantly worse than the LM judge, and a model-free name-overlap judge matches or beats it. Paper: `paper/main.tex`.

## What was run
| ID | What | Status | Output |
|---|---|---|---|
| V2 | Reproduce Agrawal et al.'s 36 AUCs from released scores | ✅ 36/36 within ±0.0005 | `results/v2_agrawal_auc.csv` |
| V1 | SelfCheckGPT on WikiBio | ✅ Unigram reproduced (85.64 vs 85.63); NLI scorer shown identical to official code (not re-benchmarked, PC-002) | `results/v1_selfcheck_unigram_only.csv` |
| E1 | Judge swap (J-LM / J-NLI / J-STR) on 6 models, 190 main topics | ✅ | `results/e1_auroc.csv`, `results/e1_deltas.csv` |
| R1 | Bing labels vs OpenAlex/Crossref (Llama-2-7B) | ✅ on 888/950 refs, κ = 0.72 (PC-006: API daily limit) | `results/r1_agreement.csv` |
| Post-hoc | E1 on Bing–oracle agreement subset (peer review M2) | ✅ judges 0.54–0.61; NLI−LM not significant | `results/posthoc_label_agreement.csv` |
| RQ2 | Fabrication vs topic familiarity | ✅ H2 not supported | `results/rq2_*.csv` |
| E2 | New generator (Qwen2.5-1.5B), detectors D1–D6 | ❌ **not run** (PC-004: 334 s/topic). Code kept: `scripts/14_e2_generate.py`, `15_e2_score.py` | 1 pilot topic in `results/raw/e2_pilot/` |

## Layout
```
configs/        protocol_v1.yaml, topic splits, FREEZE.md (protocol sign-off; git tag protocol-v1)
data/README.md  data sources, licences, download (data/raw is not committed)
docs/           01-07: review scope, gaps, mapping, novelty, brief, data sheet, protocol; upstream prompts
src/aitr/       agrawal.py (loader), judges.py, metrics.py, oracle.py, prompts.py, splits.py, seed.py, download.py
scripts/        00 env check · 01 splits · 10 V2 · 11 V1 · 12 E1 · 13 R1 · 14/15 E2 (unexecuted) · 16 RQ2 · 20 figures
tests/          unit tests (run: for t in tests/test_*.py; do python $t; done)
results/        genuine outputs only: CSVs, logs/, raw/ (cached scores and API responses), env/
paper/          main.tex, references.bib, figures/
research_decisions.md   decisions D-001..D-019 and protocol changes PC-001..PC-005
baseline_validation.md  V1/V2 targets (pre-registered) and results
env_verification_log.md environment runs (Colab T4; local Mac)
ai_assistance_log.md    AI-use record
```

## Reproduce
Python 3.13. Exact versions: `requirements-lock.txt` (Colab). The local runs used the same versions in a venv.
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python scripts/01_make_pilot_split.py          # downloads Agrawal data @bf3ef5f; pilot/main/e2 topics
for t in tests/test_*.py; do .venv/bin/python $t; done    # unit tests
.venv/bin/python scripts/10_v2_reproduce_agrawal.py      # V2, CPU, ~1 min
.venv/bin/python scripts/11_v1_reproduce_selfcheck.py --impl none   # V1 Unigram, CPU, ~8 min
.venv/bin/python scripts/12_e1_judge_swap.py --split main --judges str,nli   # E1, ~1 h on Apple MPS / CUDA
.venv/bin/python scripts/13_r1_oracle_relabel.py --split main --model llama-2-7b-chat --cache-only   # R1 from cache (888 refs)
.venv/bin/python scripts/17_posthoc_label_agreement.py   # post-hoc label-noise check
.venv/bin/python scripts/16_rq2_familiarity.py           # RQ2, network (190 searches)
.venv/bin/python scripts/20_make_figures.py              # paper figures
```
R1 and RQ2 call OpenAlex and Crossref. Set `OPENALEX_API_KEY` (free key; the keyless allowance is about 100 searches a day) and `AITR_MAILTO` (contact email for the polite pools) in the environment. Keys are never stored in this repository. All responses used in the paper are cached in `results/raw/oracle_cache.jsonl`, so reruns reproduce the labels without network access.

Not executed here, marked [TEST REQUIRED]: `scripts/11_v1_reproduce_selfcheck.py` with the default NLI path on the full benchmark (about 50 min on a T4, about 7.6 h on CPU); `scripts/14_e2_generate.py --split e2` and `15_e2_score.py --split e2`.

## Minimum reproduction checklist (for a second student)
- [ ] `01_make_pilot_split.py` prints `pilot=10 main=190 e2=100 sha256=94422a812ca0`
- [ ] All tests print `ok`
- [ ] `10_v2_reproduce_agrawal.py` ends with `V2: 36/36 cells within ±0.005`
- [ ] `results/e1_auroc.csv` matches Table 2 of the paper (re-running E1 from the cached `results/raw/e1_nli_*` needs no GPU)

## Credits and licences
This repository's own code, documentation and paper source are released under the MIT License (`LICENSE`). Third-party material keeps its original licence: the Agrawal et al. data and prompts (MIT), SelfCheckGPT (MIT), the NLI checkpoint (Apache-2.0) and the WikiBio dataset (CC-BY-SA-3.0, not redistributed here). Cached OpenAlex/Crossref responses in `results/raw/` are bibliographic metadata under those services' terms.

Data and prompts: Agrawal et al., microsoft/hallucinated-references (MIT). SelfCheckGPT (MIT) and its NLI checkpoint `potsawee/deberta-v3-large-mnli` (Apache-2.0). WikiBio hallucination dataset (CC-BY-SA-3.0). OpenAlex and Crossref APIs. AI assistance: Claude Code, logged in `ai_assistance_log.md`.
