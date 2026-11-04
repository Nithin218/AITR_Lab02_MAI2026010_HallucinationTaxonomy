# Research Brief (Gate G1; Prompt 05 output)

> **Status note (2026-11-04):** Superseded in part. RQ1 was answered by E1 on Agrawal et al.'s released data, not by a new 7B generator (PC-003, PC-004). RQ2 moved to the released data (PC-005). RQ3 was dropped with E2. V1 = Unigram reproduction + NLI implementation equivalence (PC-002). Final results: `paper/main.pdf`. See `research_decisions.md` for every decision and protocol change. The text below is kept unchanged as the original planning record.

Student: Nagireddy Nithin Kumar Reddy (MAI2026010) · Date: 2026-10-29 · Status: DRAFT, for instructor review before implementation

## Research problem
LLMs used in research workflows fabricate references (Walters & Wilder 2023; Chelli et al. 2024). Agrawal et al. (2023) showed that a model's own consistency checks can flag fabricated references for GPT-4, but the same checks were not effective for 7B–13B open models. In those checks the generator judges its own answers. It is unknown whether a generic sampling-based detector that uses an **external** NLI judge (SelfCheckGPT; Manakul et al. 2023) recovers detection ability at small scale. It is also unknown how detection depends on how well-published a topic is.

## Already demonstrated vs. what this study investigates

| Already demonstrated (cite, do not claim) | Investigated here |
|---|---|
| SelfCheck-NLI detects non-factual WikiBio sentences (AUC-PR 92.50, README) | Whether it **transfers** to reference strings from a 7B generator |
| DQ/IQ checks work for GPT-4, not Llama-2-7B/13B (Agrawal et al.) | Head-to-head: external-judge SelfCheck-NLI vs. DQ/IQ on the same 7B outputs |
| Fabrication varies by topic (Linardon et al. 2025, mental health) | Fabrication rate and detector AUROC vs. **OpenAlex publication count** in CS topics |

## Research questions

**Validation gate (before any RQ):** the reproduced SelfCheck-NLI and SelfCheck-Unigram NonFact AUC-PR on `wiki_bio_gpt3_hallucination` must fall within ±2.0 points of the published values (92.50 / 85.63, README; check against the paper, V7). If not: investigate, document, and carry the result forward as a stated validity threat.

### RQ1: Detector comparison (primary)
| Field | Specification |
|---|---|
| Input data | 5 references × T topics (T ≈ 100 ACM CCS topics, frozen in Prompt 07) from Qwen2.5-7B-Instruct (greedy), plus N = 10 samples at temperature 1.0 |
| Unit / label | One reference. Positive = **not found** (no OpenAlex or Crossref match above a title-similarity threshold fixed on a separate 10-topic pilot) |
| Comparators | D1 SelfCheck-NLI (validated baseline) · D2 SelfCheck-Unigram · D3 mean token −log p · D4 verbalised P(exists) · D5 Agrawal DQ ensemble · D6 Agrawal IQ · random |
| Independent variable | Detector |
| Dependent variable / metric | **AUROC (primary, higher = better)**; AUC-PR (secondary); 95% bootstrap CIs over topics |
| Hypothesis H1 | AUROC(D1) > AUROC(D6). Rationale: IQ depends on the weak 7B model judging its own author lists; D1 hands that judgement to a separate NLI model. |
| Decision rule | H1 supported if the paired-bootstrap 95% CI of AUROC(D1) − AUROC(D6) is entirely above 0. D1 "useful" if its AUROC CI lower bound is ≥ 0.70. |
| Contradicting observation | The CI includes or is below 0 (no advantage), or D1's AUROC CI includes 0.5 (D1 also fails at 7B, extending Agrawal's negative finding) |

### RQ2: Topic familiarity (deeper analysis)
| Field | Specification |
|---|---|
| Input data | The same references; per-topic familiarity = OpenAlex work count for the topic name (log scale), split into tertiles |
| Independent variable | Familiarity (continuous and tertile) |
| Dependent variables | (a) Fabrication rate per topic; (b) D1 and D6 AUROC per tertile |
| Metric | Logistic regression of not-found on log(work count), with clustered or bootstrap CIs by topic; per-tertile AUROC with CIs |
| Hypothesis H2 | Fabrication rate rises as familiarity falls (slope < 0) |
| Decision rule | H2 supported if the slope's 95% CI is entirely below 0. Per-tertile AUROC is **exploratory** (small n per tertile), and only descriptive claims are made. |
| Contradicting observation | Slope CI includes 0 or is positive |

### RQ3: Cost and sensitivity (efficiency)
| Field | Specification |
|---|---|
| Question | How does D1 AUROC change with N ∈ {1, 2, 5, 10} samples (subsampled, seeds fixed), and what is the GPU cost per reference for D1–D6? |
| Metric | AUROC vs. N; seconds per reference (measured) |
| Decision rule (descriptive, no hypothesis) | Report the smallest N whose AUROC is within 0.02 of N = 10 |

## Flags: overbroad or at-risk claims
- "Detectors transfer to citation fabrication" is overbroad. The findings apply to **one generator (7B), English, CS topics, OpenAlex/Crossref coverage**. Claims must say so.
- **Citation support** (does a real paper support the claim?) is **out of scope**. Only **existence** is tested.
- Ground truth is a database proxy. Report agreement with the 50-reference manual audit (Cohen's κ). Substring and near-match titles go into a separate "partial" category, excluded from the primary analysis and reported separately.
- Agrawal's Bing oracle is replaced by OpenAlex/Crossref, so absolute rates are **not directly comparable** with theirs.
- Any matching threshold is fixed on the pilot topics only. Pilot topics are excluded from the main set to avoid leakage.

## Dataset / setup
- Baseline: `potsawee/wiki_bio_gpt3_hallucination` (CC-BY-SA-3.0)
- Extension: self-generated references from ACM CCS topic names; labels from OpenAlex (no key needed) and Crossref (polite pool)

## Baseline / comparators
SelfCheckGPT (MIT, pip) plus DQ/IQ re-implemented from Agrawal et al. (MIT repo; prompts copied verbatim, with any deviation logged).

## Primary metric and controls
AUROC over not-found references. Held constant across detectors: generator, weights, quantisation, the main answers, the sample set, seeds, decoding settings and topic list.

## Compute and risks
Colab T4, about 4–6 GPU hours, checkpointed to Drive. Risks: session limits, OpenAlex coverage of books, prompt sensitivity. See `04_feasibility_novelty.md` §7 for fallbacks.

## First executable milestone (Day 3)
A Colab notebook that (1) prints GPU and library versions, (2) runs `SelfCheckNLI` on 5 WikiBio passages and prints sentence scores, (3) loads Qwen2.5-7B in 4-bit and times one 5-reference generation, and (4) queries OpenAlex for 10 real and 10 invented titles.

## Key references (to be checked from the originals)
Manakul et al. 2023 (SelfCheckGPT) · Agrawal et al. 2023/2024 · Kadavath et al. 2022 · Linardon et al. 2025 · Walters & Wilder 2023
