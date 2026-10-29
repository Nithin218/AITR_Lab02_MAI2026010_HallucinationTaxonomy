# Dataset and Baseline Validation Sheet (Prompt 06 output)

Research question (from `05_research_brief.md`): for small open-weight LLMs generating references on ACM CCS topics, does a detector with an **external** judge (NLI) detect fabricated references better than LM-judged consistency checks? How does detection vary with topic familiarity?
Domain: NLP / hallucination detection. There is **no training**: every detector is zero-shot, so there is no train/test split in the usual sense (see §4).

Checks run 2026-10-29. "Executed" means the AI assistant downloaded or loaded the files in a scratch directory and inspected them. No metrics were computed.

---

## 1. Datasets

### DS-1 (primary, extension): Agrawal et al. released results, `microsoft/hallucinated-references/data/`
| Item | Finding | Label |
|---|---|---|
| Official source | github.com/microsoft/hallucinated-references (paper arXiv:2305.18248) | VERIFIED |
| Licence | MIT (repo) | VERIFIED (repo page); whether the data has a separate licence is UNKNOWN |
| Files | `acm_ccs_200.titles`; `gpt-4_results.csv`; `gpt-3.5-turbo_results.csv`; `text-davinci-003_results.csv`; `llama-2-{7b,13b,70b}-chat.csv`; `expert_annotation_study/{expert_A..D, ground_truth}.csv` | VERIFIED (all 7 data files return HTTP 200) |
| `acm_ccs_200.titles` | 200 lines, one topic each (e.g. "Computer forensics: System forensics") | VERIFIED (executed) |
| `llama-2-7b-chat.csv` | 1,000 rows; 200 distinct `concept` values, all in `acm_ccs_200`; columns `gen_title, concept, model_answer_main_query, bing_return, bing_return_results, IQ_full_ans1-3, IQ_ans1-3, IQ_llama_prob, IQ_llama_ans_list, DQ{1,2,3}_ans_sample, DQ{1,2,3}_prob_sample` | VERIFIED (executed) |
| `gpt-4_results.csv` | 1,000 rows; 200 topics; **different schema**: `title` (topic), `model_ans_1-3` (author samples), `neural_ans{1-4}_list/_prob` | VERIFIED (executed) |
| Label check | `bing_return == False`: Llama-2-7B 683/1000 = 68.3%; GPT-4 468/1000 = 46.8%. Both match the paper's reported hallucination rates. | VERIFIED (executed; data integrity only, not an outcome) |
| Benchmark protocol | Label = exact-title Bing search; hallucinated if no results. IQ = 3 author samples judged by the LM. DQ = fraction of "yes" over 10 samples. | VERIFIED (paper HTML) |
| Compute | CSVs total ~16 MB. Re-scoring with NLI on a T4 is INFERRED to take under 30 min per model. | INFERRED |

**Pitfalls (DS-1):**
1. Column names differ between the GPT and Llama files. Which `neural_ans{k}` is IQ and which are DQ1–3 in the GPT files is **UNKNOWN**. Resolve it from `code/metrics.ipynb` before analysis.
2. `bing_return` is a string (`"True"`/`"False"`). Hallucinated = `"False"`. Getting this backwards flips every AUROC.
3. 6 Llama-7B rows have an empty `IQ_ans1` (VERIFIED). Decide now: keep them and score as maximally inconsistent, or exclude and report the count.
4. `IQ_ans*` contain chat boilerplate ("Sure, here are the author names…"). Normalising author names needs a fixed rule set, designed on pilot topics only.
5. Some generated titles echo the topic name (2 exact echoes in Llama-7B, VERIFIED). Bing finds those trivially. Report them; do not drop them silently.
6. Labels date from 2023. Re-checking titles against today's databases can disagree, because generic titles may since have been published.

### DS-2 (baseline validation): `potsawee/wiki_bio_gpt3_hallucination`
| Item | Finding | Label |
|---|---|---|
| Source | HF dataset; paper arXiv:2303.08896 (v3, 11 Oct 2023) | VERIFIED |
| Structure | 238 rows, `evaluation` split: `gpt3_text`, `wiki_bio_text`, `gpt3_sentences`, `annotation` ∈ {accurate, minor_inaccurate, major_inaccurate}, `wiki_bio_test_idx`, `gpt3_text_samples` (20 samples) | VERIFIED (dataset card) |
| Licence | CC-BY-SA-3.0 | VERIFIED |
| Protocol | Sentence-level AUC-PR for NonFact / Factual; passage-level PCC | VERIFIED (README); **the NonFact label mapping is UNKNOWN** (check the paper) |
| Compute | NLI: ~1.9k sentences (INFERRED) × 20 samples. 20–60 min on a T4. | INFERRED |

**Pitfalls (DS-2):** NonFact mapping (does minor_inaccurate count?); the README may report a later package version than the paper; the "NonFact*" variant may exist in the paper but not the README.

### DS-3 (new generations, E2): self-generated with Qwen2.5-7B-Instruct
| Item | Finding | Label |
|---|---|---|
| Topics | The same `acm_ccs_200.titles`, so results are comparable with DS-1 | VERIFIED (file exists) |
| Generator | Qwen/Qwen2.5-7B-Instruct, Apache-2.0, 7.61B params, not gated | VERIFIED (model card); 4-bit T4 fit UNKNOWN |
| Labels | OpenAlex (no key, 10 req/s) and Crossref (polite pool, `mailto`) | VERIFIED (docs) |
| Pitfalls | OpenAlex/Crossref under-cover books (Agrawal asked for books *and* articles); fuzzy-match thresholds; generic titles matching by chance; Qwen2.5 may have seen Agrawal's public CSVs during training (contamination, INFERRED) | — |

---

## 2. Baselines / comparators

| ID | Method | Official source | Access | Role |
|---|---|---|---|---|
| **B1** | SelfCheck-NLI: `SelfCheckNLI`, default checkpoint `potsawee/deberta-v3-large-mnli` (Apache-2.0); score = P(contradiction) from a 2-way softmax, averaged over samples | `selfcheckgpt` (MIT) | VERIFIED (source code `utils.py`, `modeling_selfcheck.py`; model card) | Validated detector; reproduced on DS-2 |
| **B2** | Agrawal IQ (LM-judged author consistency) and DQ1–3 | `microsoft/hallucinated-references` (MIT) | VERIFIED: released scores exist in DS-1 | Published comparator; reproduced from released scores |
| B3 | SelfCheck-Unigram: `SelfCheckNgram(n=1)`; unbounded score | `selfcheckgpt` | VERIFIED (source) | Cheap control |
| B4 | String-overlap judge: 1 − mean pairwise Jaccard of normalised author surnames | Own code (no model) | — | Non-learned control for the judge swap |
| B5 | Random / prevalence | — | — | Floor |

Pitfalls: the label-index assumption in `SelfCheckNLI` (index 1 = contradiction) matches the model card example (VERIFIED). The DeBERTa input order is premise/hypothesis; record which answer is premise. `SelfCheckNgram` scores are unbounded, which is fine for AUROC but not for averaging with other scores.

---

## 3. Simplest credible baseline

**B2 reproduced from released data.** Recompute IQ and DQ AUROC from the scores already in DS-1 against `bing_return`, for all six models. This needs no GPU and no generation, and must match the paper's reported values (GPT-4: DQ1 0.887, IQ 0.878). Every other paper number must first be copied from the PDF, V11.

**B1 reproduced on DS-2** validates our NLI scorer before it is used on references.

---

## 4. Evaluation design (zero-shot detectors, no training)

- There is no model training, so a train/test split is **not required for the detectors**. AUROC is threshold-free.
- **Some things are tuned** (author-name normalisation rules, title-match threshold τ, prompt formatting). These are fixed on a **pilot partition**: 10 topics drawn with seed 2026 from `acm_ccs_200`. The pilot is used **only** for those choices and is **excluded** from all reported results. Main set = the 190 remaining topics.
- Unit of analysis = one generated reference. References cluster within topics, so CIs use a **topic-level cluster bootstrap**.
- Manual audit: 50 references from the main set, stratified 25 found / 25 not-found by the automatic label, checked in Google Scholar. Report Cohen's κ against the automatic label.

## 5. Leakage and validity risks
| Risk | Level | Mitigation |
|---|---|---|
| Thresholds or rules tuned on evaluation topics | Topic | Pilot/main split above |
| Same topic's references in both pilot and main | Topic | Split by topic, not by reference |
| Choosing the NLI judge's aggregation after seeing AUROC | Analysis | Aggregation fixed in the protocol (07) before scoring |
| Ground-truth time shift (Bing 2023 vs. OpenAlex 2026) | Time | Report agreement as a robustness analysis; never mix oracles within one comparison |
| Generator contamination with DS-1 | Document | State as a threat; compare Qwen's titles with DS-1 titles (exact-overlap count) |
| Author names in IQ samples copied from the main answer | Document | Agrawal's IQ prompts ask in separate sessions (VERIFIED in paper text); keep that |

## 6. Independent-verification checklist (student)
- [ ] Open the Agrawal paper's Figure/Table with per-model AUCs; copy all IQ/DQ numbers into `baseline_validation.md` (V11)
- [ ] Read `code/metrics.ipynb` and map `neural_ans{1-4}` → IQ/DQ1/DQ2/DQ3 for the GPT files
- [ ] Open the SelfCheckGPT paper results table: NLI value, NonFact definition (V7)
- [ ] Confirm the DeBERTa label order by running 2 hand-made examples (one entailment, one contradiction)
- [ ] Confirm that Qwen2.5-7B-Instruct loads in 4-bit on your Colab GPU
- [ ] Re-download the DS-1 files yourself and check that row counts match this sheet (1,000 per model)
