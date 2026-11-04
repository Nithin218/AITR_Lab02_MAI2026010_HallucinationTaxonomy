# Baseline Validation Report (Prompt 10)

Status: **NOT STARTED** (only setup facts below). This is filled in only from real runs. Each item is labelled VERIFIED / UNCERTAIN / NEEDS CORRECTION.

## V1: SelfCheckGPT on WikiBio
| Check | Published | Ours | Status |
|---|---|---|---|
| Dataset size | 238 passages (card) | 238 passages, 1,908 sentences (env Run 1) | VERIFIED |
| NonFact definition (minor+major inaccurate?) | README random NonFact = 72.96, Factual = 27.04 | (761+631)/1908 = 72.96%; 516/1908 = 27.04% | VERIFIED by arithmetic: NonFact = minor + major. Confirm the wording in the paper. |
| NLI label index 1 = contradiction | source code assumption | 0.0002 vs 0.9997 on hand-made pairs | VERIFIED |

### V1 pre-registration (committed 2026-11-04, before the first run)
- Script: `scripts/11_v1_reproduce_selfcheck.py`. The official `SelfCheckNLI` / `SelfCheckNgram(n=1)` classes are used unchanged, with all 20 samples per passage on all 238 passages.
- Targets (selfcheckgpt README): NLI NonFact 92.50 / Factual 66.08 / PCC 74.14; Unigram 85.63 / 58.47 / 64.71.
- **Pass criterion: NonFact AUC-PR within ±2.0 points** (research brief). Factual and PCC are reported, not gated.
- Unigram target = the `max_neg_logprob` variant (INFERRED from the paper's "Unigram (max)"). The `avg` variant is reported too.
- PCC passage score = mean of sentence labels with accurate = 0, minor = 0.5, major = 1 (INFERRED mapping). If only PCC misses, the mapping is the first suspect, not the scorer.

### V1 implementation check: batched NLI vs. official `SelfCheckNLI` (2026-11-04, local Mac MPS)
- Official class run on passages 0–2 (`--impl official --limit 3`): `results/raw/v1_selfcheck_scores_official.jsonl`
- Batched scorer on the same passages: **max |diff| = 2.03e-7, mean 7.19e-8** over 25 sentence scores. **VERIFIED: equivalent.**
- Speed on the Apple GPU: about 4 s per pair for whole passages, so the full V1-NLI is impractical locally (PC-001). Unigram runs on CPU.
### V1 results (2026-11-04, local Mac; `python scripts/11_v1_reproduce_selfcheck.py --impl none`)
| Method | NonFact AUC-PR | Factual AUC-PR | PCC | Target (README) | NonFact diff | Pass (±2.0) |
|---|---|---|---|---|---|---|
| SelfCheck-Unigram (max) | **85.64** | 58.50 | 64.72 | 85.63 / 58.47 / 64.71 | +0.01 | **VERIFIED** |
| SelfCheck-Unigram (avg) | 81.54 | 41.97 | 37.81 | (not a target) | — | — |

All 238 passages and 1,908 sentences. Every Unigram figure matches the README to ≤ 0.03. This also confirms the two INFERRED mappings: NonFact = minor + major, and PCC passage score with accurate 0 / minor 0.5 / major 1. File: `results/v1_selfcheck_unigram_only.csv`.

- **Scope of V1 (PC-002, student decision):** Unigram is reproduced on the full benchmark. NLI is validated by equivalence to the official implementation only. The published NLI AUC-PR (92.50) is **not re-measured**, and the paper states this as a limitation.
| SelfCheck-NLI NonFact AUC-PR | 92.50 (README) / _paper_ | | |
| SelfCheck-Unigram (max) NonFact AUC-PR | 85.63 (README) / _paper_ | | |
| Factual AUC-PR, PCC | README table | | |

## V2: Agrawal et al. from released scores

### Pre-registered before computing (committed 2026-10-30, before any V2 number was produced)
- **Published values:** text embedded in the ROC figures of arXiv:2305.18248v3 (Fig. 5 a–c, Fig. 9 a–c), extracted with `pypdf`. The panel label follows its values on each page. The section 6.1 text gives GPT-4 IQ 0.878 and DQ1 0.887, while Fig. 5c shows 0.877 and 0.885. The **figure values are the target**, because the released `metrics.ipynb` produces those figures.
- **Method replicated** (from `metrics.ipynb` + `roc-utils` 0.2.2 source):
  - Score = 1 − released prob; y = 1 if `bing_return` is False.
  - DQ = mean of the DQ1–3 scores; IQ+DQ = mean(IQ, DQ).
  - 100 bootstrap resamples, a **fresh `np.random.RandomState(42)` per method**, indices `rng.randint(0, n, n)`; resamples lacking a class are skipped.
  - Per-resample trapezoid AUC; the reported value is the mean of the resample AUCs, **auto-flipped to 1 − AUC if the mean ROC curve's AUC < 0.5**.
  - 13B file as released (D-014).
- **Pass criterion:** |ours − published| ≤ **0.005** for every cell. Rounding alone allows 0.0005; the rest covers interpolation and float differences. Any cell outside that band is investigated, not tuned.
- `roc-utils` itself cannot be imported with current SciPy (`from scipy import interp` fails), so the method is reimplemented and the difference is logged as a deviation.

| Model | IQ | DQ1 | DQ2 | DQ3 | DQ | IQ+DQ |
|---|---|---|---|---|---|---|
| GPT-4 (Fig 5c) | 0.877 | 0.885 | 0.842 | 0.859 | 0.915 | 0.927 |
| ChatGPT / gpt-3.5-turbo (5b) | 0.782 | 0.540 | 0.610 | 0.528 | 0.624 | 0.792 |
| Llama-2-70B (5a) | 0.658 | 0.504 | 0.631 | 0.572 | 0.658 | 0.702 |
| GPT-3 / text-davinci-003 (9c) | 0.695 | 0.555 | 0.588 | 0.534 | 0.621 | 0.710 |
| Llama-2-13B (9b) | 0.657 | 0.500 | 0.686 | 0.546 | 0.696 | 0.723 |
| Llama-2-7B (9a) | 0.517 | 0.500 | 0.574 | 0.501 | 0.573 | 0.575 |

### Results (run 2026-10-30, local Mac; numpy 2.1.3, scikit-learn 1.6.1, the same versions as `requirements-lock.txt`)
Command: `python scripts/10_v2_reproduce_agrawal.py` → `results/v2_agrawal_auc.csv`.

**36/36 cells within ±0.005; max |diff| = 0.0005. V2 is VERIFIED.** No method was auto-flipped. All 100 resamples were used in every cell.

| Model | IQ | DQ1 | DQ2 | DQ3 | DQ | IQ+DQ |
|---|---|---|---|---|---|---|
| GPT-4 | 0.8768 | 0.8854 | 0.8423 | 0.8590 | 0.9149 | 0.9274 |
| gpt-3.5-turbo | 0.7818 | 0.5399 | 0.6098 | 0.5283 | 0.6239 | 0.7922 |
| Llama-2-70B | 0.6577 | 0.5036 | 0.6308 | 0.5719 | 0.6583 | 0.7020 |
| text-davinci-003 | 0.6947 | 0.5553 | 0.5880 | 0.5344 | 0.6211 | 0.7104 |
| Llama-2-13B (as released) | 0.6568 | 0.5000 | 0.6858 | 0.5460 | 0.6961 | 0.7235 |
| Llama-2-7B | 0.5172 | 0.5000 | 0.5742 | 0.5010 | 0.5729 | 0.5747 |

(Bootstrap means. Point AUCs are in the CSV.)

Findings from V2:
1. **The text-vs-figure discrepancy is explained.** GPT-4 point AUCs are IQ 0.8781 and DQ1 0.8868. These round to the section 6.1 text values (0.878, 0.887), while the bootstrap means match the figure (0.877, 0.885). So the text reports point AUCs and the figures report bootstrap means. INFERRED, but it matches both cells exactly.
2. **The 13B duplication (D-014) barely moves the published numbers.** Deduplicated (n = 800): IQ 0.6602 (+0.003), DQ2 0.6876, DQ3 0.5497, DQ 0.6964, IQ+DQ 0.7294 (+0.006). The defect does not change any conclusion of the paper.
3. **The paper's small-model claim reproduces.** Llama-2-7B IQ = 0.517 and DQ1 = 0.500 (constant "yes"). This is the gap E1 tests.

## Deviations from the original setup
| # | Deviation | Acceptable? Why |
|---|---|---|
| 1 | V2: `roc-utils` 0.2.2 is reimplemented (`src/aitr/metrics.py`) because it cannot be imported with current SciPy (`scipy.interp` removed) | Yes. Same RNG, sampling, skip rule, rounding and auto-flip, and 36/36 values match to ≤ 0.0005 |
| 2 | V2: per-resample AUC from `sklearn.roc_auc_score` on 8-decimal-rounded scores instead of roc-utils' own trapezoid | Yes. Equivalent tie handling (`tests/test_metrics.py::test_near_ties_are_merged`); confirmed by the match |

## Additional checks required
