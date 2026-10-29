# Baseline Validation Report (Prompt 10)

Status: **NOT STARTED** (only setup facts below). This is filled in only from real runs. Each item is labelled VERIFIED / UNCERTAIN / NEEDS CORRECTION.

## V1: SelfCheckGPT on WikiBio
| Check | Published | Ours | Status |
|---|---|---|---|
| Dataset size | 238 passages (card) | 238 passages, 1,908 sentences (env Run 1) | VERIFIED |
| NonFact definition (minor+major inaccurate?) | README random NonFact = 72.96, Factual = 27.04 | (761+631)/1908 = 72.96%; 516/1908 = 27.04% | VERIFIED by arithmetic: NonFact = minor + major. Confirm the wording in the paper. |
| NLI label index 1 = contradiction | source code assumption | 0.0002 vs 0.9997 on hand-made pairs | VERIFIED |
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

### Results
_Filled in by `scripts/10_v2_reproduce_agrawal.py` → `results/v2_agrawal_auc.csv`._

## Deviations from the original setup
| # | Deviation | Acceptable? Why |
|---|---|---|

## Additional checks required
