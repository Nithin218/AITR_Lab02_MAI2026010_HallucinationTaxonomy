# Literature-to-Experiment Map (Prompt 03 output)

Constraints (from the student, 2026-10-29): Google Colab (free tier assumed), Claude Pro subscription, 15 days, solo or group not stated.
Candidate gaps: G1, G2 and G4 from `02_gap_matrix.md`.

Access checks done on 2026-10-29 (web pages fetched by the AI assistant; the student should re-open each link):

| Resource | Finding | Label |
|---|---|---|
| SelfCheckGPT code: github.com/potsawee/selfcheckgpt | MIT licence; `pip install selfcheckgpt`; classes `SelfCheckNLI`, `SelfCheckBERTScore`, `SelfCheckMQAG`, `SelfCheckNgram`, `SelfCheckLLMPrompt`, `SelfCheckAPIPrompt` | VERIFIED (README fetched) |
| SelfCheckGPT data: huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination | 238 passages, `evaluation` split; sentence labels `accurate` / `minor_inaccurate` / `major_inaccurate`; **20 pre-generated GPT-3 samples per passage are included**; CC-BY-SA-3.0 | VERIFIED (dataset card fetched) |
| SelfCheckGPT README results (AUC-PR, NonFact) | SelfCheck-NLI 92.50; SelfCheck-Unigram 85.63; GPT-3 Avg(−logP) 83.21; random 72.96 | VERIFIED as README figures. **UNKNOWN** whether they match the EMNLP paper's table (check V7). |
| OpenAlex API | No key needed for basic queries (10 req/s); a free key gives a 10× daily budget and is needed for cursor paging | VERIFIED (help.openalex.org fetched) |
| Crossref REST API | Pages fetched did not state key or rate-limit rules | **UNKNOWN**. Student must check the Crossref "Access and authentication" docs. |
| Claude Pro as an experimental generator | Claude Pro is a chat subscription. It does not include Anthropic API credits, so it cannot be called from Colab code. | INFERRED (student should confirm in their account; API access is billed separately) |
| Open-weight generator on Colab (e.g. Qwen2.5-7B-Instruct, 4-bit, T4 GPU) | Licence and VRAM fit are from memory | **UNKNOWN**. Verify the model card licence and run a Colab load test. |

**Consequence of the Claude Pro finding:** the LLM that generates references must be an **open-weight model running in Colab**. This also helps the study: fixed weights and seeds make it reproducible, and token log-probabilities become available as an extra detector. Claude Pro is used as the *coding/writing assistant*, and that use is logged in `ai_assistance_log.md`.

---

## 1. Three published methods from the review, mapped

| | M1 SelfCheckGPT (Manakul et al. 2023) | M2 FActScore (Min et al. 2023) | M3 Semantic entropy (Kuhn et al. 2023) |
|---|---|---|---|
| Review location | l.185, Table 2 | l.188, Table 2 | l.137, l.194 |
| Algorithm | Sample N extra responses; score each sentence of the main response by its agreement with the samples (NLI, BERTScore, n-gram, QA, LLM prompt) | Split long text into atomic facts; check each against a knowledge source (Wikipedia) | Sample answers, cluster them by bidirectional entailment, take entropy over the clusters |
| Dataset | wiki_bio_gpt3_hallucination, 238 passages (VERIFIED) | Biographies (UNKNOWN details) | QA datasets, e.g. TriviaQA/CoQA (UNKNOWN details) |
| Code/model access | VERIFIED: MIT, pip package. Pre-computed samples mean **no generator is needed to reproduce the baseline**. | UNKNOWN: code exists from memory; needs an LLM for fact decomposition plus a Wikipedia dump | UNKNOWN: code exists from memory; needs generation with log-probs |
| Metric | AUC-PR (NonFact, Factual), PCC | FActScore (factual precision) | AUROC |
| Known limitation (review) | Costly; stable ≠ true (l.185) | Needs a knowledge source (l.188) | Uncertainty ≠ hallucination (l.194) |
| Compute on Colab | NLI scorer (DeBERTa-large class, from memory): 238 × ~7 sentences × 20 samples ≈ 33k NLI pairs. INFERRED to take under 1 h on a T4. | Heavy: LLM decomposition and retrieval | Moderate: needs a generator |
| Verdict | **Best baseline**: reproducible in 1–2 days | Too heavy for 15 days | Possible as an extra detector; optional |

---

## 2. Three candidate study designs

| | (A) Reproduction + extension | (B) Controlled comparison | (C) Targeted modification |
|---|---|---|---|
| Study | Reproduce SelfCheck-NLI on WikiBio, then apply it to **generated references** under a new condition | Compare detectors (self-consistency, log-prob, verbalised confidence) on generated references, with database-derived ground truth | Change one SelfCheck component, e.g. replace sentence-level NLI with field-level matching of title, authors and year |
| Smallest working baseline | `SelfCheckNLI` on the 238 WikiBio passages | SelfCheck-NLI on references | SelfCheck-NLI as in (A) |
| Testable question | Does SelfCheck-NLI's detection quality carry over from biography sentences to bibliographic references? | Which cheap signal best predicts that a reference does not exist? | Does reference-aware matching beat generic NLI? |
| Resource burden | Low (baseline), then moderate (generation) | Moderate | Moderate, plus design risk |
| Evidence needed | Reproduced AUC-PR near 92.50; AUC-PR/AUROC on the reference set | Labelled reference set; AUROC per detector | Ablation that isolates the change |

**These are not exclusive.** The recommended study (§3) is **A + B**: reproduction gives the verified baseline the G2 gate requires, and the controlled comparison answers the gap. Design (C) is a stretch goal only.

---

## 3. Recommended minimum viable experiment (15 days, Colab)

**Working title:** *Do zero-resource hallucination detectors transfer to citation fabrication? A reproduction-and-extension study of SelfCheckGPT.*

1. **Baseline (Pathway A, verified):** reproduce SelfCheck-NLI and SelfCheck-Unigram on `wiki_bio_gpt3_hallucination` using the pre-computed samples. Compare against README/paper AUC-PR.
2. **New condition:** an open-weight LLM in Colab is asked for K references on each of T research topics. It produces one greedy main answer plus N sampled answers (temperature 1.0, as in the dataset card).
3. **Ground truth:** each generated reference is looked up in OpenAlex (and Crossref, if its access is confirmed) using title similarity plus author/year matching. Labels are *exists-correct* / *exists-metadata-error* / *not-found*. A **manual audit** checks a random subset to measure label noise.
4. **Detectors compared (two or more comparators, as required):**
   - D1 SelfCheck-NLI (the validated baseline)
   - D2 SelfCheck-Unigram (cheap, no model)
   - D3 average token −log p of the reference string (white-box; mirrors the README's "Avg(−logP)" row)
   - D4 verbalised confidence ("probability this reference exists"), which covers gap G3
   - Reference point: random baseline (equal to the positive rate)
5. **Deeper analysis (G2):** split topics by OpenAlex `works_count` (familiarity) and report fabrication rate and detector AUROC per stratum. Also report cost (seconds and samples per reference) and sensitivity of D1 to N, i.e. N ∈ {2, 5, 10} subsampled from the same generations at no extra cost.
6. **Primary metric (to be frozen in Prompt 07):** AUROC for detecting *not-found* references. Secondary metrics: AUC-PR and fabrication rate with 95% confidence intervals.

Indicative size (INFERRED, to be set in Prompt 07 after a timing test): T ≈ 40–60 topics, K = 5 references, N = 10 samples, giving ≈ 200–300 main references and ≈ 2,200–3,300 generations. Whether this fits a free Colab session is **UNKNOWN**. A timing test is required, and the fallback is a smaller model or a smaller N.

---

## 4. Mandatory bridge

| Original paper (review) | → Method selected | → Executable baseline | → Planned test |
|---|---|---|---|
| Manakul et al. 2023 (l.185) | SelfCheckGPT NLI/Unigram | `selfcheckgpt` on wiki_bio_gpt3_hallucination | Transfer to generated references (D1, D2) |
| Kadavath et al. 2022; Dahl et al. 2024 (l.137, 296) | Self-reported confidence | Prompted P(exists) | D4 calibration and AUROC |
| Walters & Wilder 2023; Chelli et al. 2024 (l.77, 197) | Database-backed citation audit | OpenAlex/Crossref lookup | Ground-truth labels plus manual audit |
| Linardon et al. 2025; Kalai & Vempala 2024 (l.293, 302) | Topic familiarity / long tail | OpenAlex works_count strata | Fabrication rate and AUROC by stratum |

---

## 5. What you must personally verify before coding

1. Open the SelfCheckGPT **paper** and record its NLI result. Note whether it matches the README's 92.50 (the README may include later variants).
2. Check how the paper turns the three labels into binary NonFact (from memory, minor and major inaccurate count as non-factual; **UNKNOWN**).
3. Check which NLI checkpoint `SelfCheckNLI` loads by default and its licence.
4. Check the licence and Colab VRAM fit of the chosen generator (e.g. Qwen2.5-7B-Instruct vs. a 3B fallback).
5. Check Crossref API access rules, or decide to use OpenAlex only.
6. In Linardon et al., check how they defined topic familiarity, so your strata are justified.
7. Confirm with the instructor that A + B counts as one pathway; otherwise label it Pathway A.
