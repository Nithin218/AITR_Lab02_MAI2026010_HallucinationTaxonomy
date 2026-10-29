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
| Model | Method | Published AUC | Ours (released scores) | Status |
|---|---|---|---|---|
| GPT-4 | DQ1 | 0.887 | | |
| GPT-4 | IQ | 0.878 | | |
| … | … | _copy from paper_ | | |

## Deviations from the original setup
| # | Deviation | Acceptable? Why |
|---|---|---|

## Additional checks required
