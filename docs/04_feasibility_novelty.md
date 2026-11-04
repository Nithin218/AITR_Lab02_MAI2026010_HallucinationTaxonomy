# Feasibility, Novelty and Resource Access (Prompt 04 output)

> **Status note (2026-11-04):** Superseded in part. The compute estimates here were too low (measured: D-013, PC-004). Oracle access needed an OpenAlex key and a contact email (D-019). R1 hit the API's daily limit (PC-006). The novelty matrix and the Agrawal et al. findings remain valid. See `research_decisions.md` for every decision and protocol change. The text below is kept unchanged as the original planning record.

Candidate study (from D-006): reproduce SelfCheckGPT on WikiBio, then transfer it to fabricated-citation detection with an open-weight generator in Colab. Compare it with other cheap signals and stratify by topic familiarity.

Search done 2026-10-29 by the AI assistant: 2 web searches plus 9 page fetches. This is **not** a systematic search. The student must run the search strings in §1 and add what they find.

---

## 1. Novelty-verification plan

### Search strings (Google Scholar, Semantic Scholar, ACL Anthology, arXiv)
1. `"hallucinated references" detection language model`
2. `"citation hallucination" OR "fabricated citations" detection LLM`
3. `"reference hallucination" consistency OR "self-consistency" OR SelfCheckGPT`
4. `SelfCheckGPT citation OR bibliographic OR reference`
5. `"verbalized confidence" OR "P(True)" references hallucination`
6. `citation fabrication "topic familiarity" OR "long-tail" OR popularity LLM`
7. `bibliographic verification OpenAlex OR Crossref LLM-generated references`
8. Forward citations ("cited by") of Manakul et al. 2023 and Agrawal et al. 2023 (below), filtered on "reference" or "citation"

### Neighbouring work to inspect
- (a) Sampling- or consistency-based hallucination detectors applied to references
- (b) Retrieval-grounded citation checkers (external database or web)
- (c) Calibration and self-knowledge of LLMs about bibliographic facts
- (d) Entity popularity / long-tail effects on factuality
- (e) Citation-URL and link-rot audits
- (f) Benchmarks of LLM-generated references with existence labels

---

## 2. Key finding: the closest prior work was not in the review

**Agrawal, Suzgun, Mackey & Kalai, "Do Language Models Know When They're Hallucinating References?"** arXiv:2305.18248 (v3, 20 Mar 2024); Findings of EACL 2024 according to search results. Code: github.com/microsoft/hallucinated-references (MIT).

What the full text says (arXiv HTML v3, fetched 2026-10-29; **VERIFIED** as fetched, but the student must read it):
- Models: GPT-3 (text-davinci-003), ChatGPT, GPT-4, **Llama-2-Chat 7B / 13B / 70B**
- 200 of 543 ACM CCS topics × 5 titles = 1,000 references per model
- Ground truth: exact-title Bing search (hallucinated if no results). Manual check of 10% of GPT-4 references gave κ 0.96–1.0.
- Hallucination rates: GPT-4 46.8%, ChatGPT 59.6%, GPT-3 73.6%, L2-70B 66.2%, L2-13B 76.7%, **L2-7B 68.3%**
- Detectors: direct queries DQ1–3 (fraction of "yes" over 10 samples); indirect query IQ (consistency of author lists across 3 sessions, judged by the LM itself); IQ+DQ ensemble
- Metric: ROC AUC and FDR. GPT-4: DQ1 0.887, IQ 0.878. The ensemble was best for every LM. **"The methods were not effective for L2-7B and L2-13B."**
- Stated limitations: web search is only a proxy; titles are not strictly binary (substring/superstring cases); prompt sensitivity; computer-science topics only; possible bias in author-name matching

**What this means:**
1. The original G1 idea ("can zero-resource signals detect fabricated references?") **has already been demonstrated** for GPT-family models. Agrawal et al. is the first paper to cite, not a gap.
2. Agrawal et al. found consistency checks **fail at 7B–13B scale**. They used the generator LM itself to judge consistency. SelfCheck-NLI instead hands the judgement to a separate NLI model. That gives a specific, testable and literature-grounded question (§4).
3. Your review (T20) does not cite this paper. Paper 2 should acknowledge it in Related Work. That is normal: the review was not designed to be exhaustive.

Other recent neighbouring work found (abstracts only, **not** read in full):
- Misra & Udandarao, "Detecting Citation Hallucinations in Large Language Model Outputs (Student Abstract)", AAAI 2026, doi:10.1609/aaai.v40i48.42257. A hybrid retrieval + fuzzy-match + LLM verification pipeline, plus a benchmark labelled valid / partially valid / hallucinated. (Fetched.)
- Rao, Wong & Callison-Burch, "Detecting and Correcting Reference Hallucinations in Commercial LLMs and Deep Research Agents", arXiv:2604.03173 (Apr 2026). URL-level audit; 3–13% of citation URLs fabricated; `urlhealth` tool. (Fetched.)
- "CiteCheck: Retrieval-Grounded Detection of LLM Citation Hallucinations in Scientific Text", alphaxiv 2605.27700. Seen only in a search snippet. **UNKNOWN**: not fetched, so authors and details are unconfirmed.

---

## 3. Novelty matrix

| Published method | VERIFIED contribution (source) | Limitation | Proposed difference in this study | Evidence still missing |
|---|---|---|---|---|
| SelfCheckGPT (Manakul et al., EMNLP 2023) | Sampling-based black-box detector; WikiBio GPT-3 benchmark; NLI variant AUC-PR 92.50 NonFact (GitHub README, fetched) | Evaluated on biographies; stable-but-wrong outputs are missed (review l.185) | Apply the validated NLI/unigram scorers to generated **reference strings** | Paper-table numbers vs. README (V7); NonFact label mapping |
| Agrawal et al. (arXiv 2305.18248; EACL Findings 2024) | DQ/IQ consistency checks detect hallucinated references for GPT-4 (AUC ≈ 0.88); not effective for Llama-2 7B/13B (arXiv HTML v3, fetched) | Bing as ground truth; CS only; LM judges its own consistency; Llama-2 era models | (i) Test whether an **external NLI judge** (SelfCheck-NLI) succeeds where LM-judged IQ failed at 7B. (ii) Newer 7B generator. (iii) Scholarly-database ground truth. (iv) Familiarity strata. | Agrawal's exact prompts (repo `code/`); whether DATA.md includes titles and labels; Venue (EACL Findings) from search snippets only, unconfirmed |
| Kadavath et al. 2022 (review l.137) | In review: confidence carries information "under some settings" | Large proprietary models; general QA | Verbalised P(exists) for a 7B model on references (detector D4) | V6: settings and model sizes |
| Linardon et al. 2025 (review l.302) | In review: topic-dependent fabrication in mental health | One domain; familiarity operationalisation **UNKNOWN** | Objective familiarity = OpenAlex work count per topic, in CS topics | V3: their definition and effect size |
| Misra & Udandarao 2026 (AAAI SA) | Retrieval + fuzzy + LLM verifier beats exact matching (abstract, fetched) | Uses external retrieval; a 2-page student abstract | Not a competitor. It is a **ground-truth method** like ours. Cite as related. | Full text; benchmark availability |
| Rao et al. 2026 | URL-level fabrication rates across 10 models and agents (abstract, fetched) | URLs, not titles | Out of scope (titles, not URLs); cite as related | — |

**Novelty verdict (INFERRED, course level):** the study **does not** claim to be first to detect hallucinated references. The defensible contribution is a controlled test of whether a generic, externally-judged detector (SelfCheck-NLI) closes the small-model gap that Agrawal et al. reported for LM-judged consistency checks. Database ground truth and familiarity strata are added. This status holds only until the student's own search (§1) is complete.

---

## 4. Narrowed experimental question

> For a 7B open-weight model asked for references on ACM CCS topics, does SelfCheck-NLI detect not-found references better than Agrawal et al.'s DQ/IQ checks, and does detection worsen for less-published topics?

Using **ACM CCS topics** follows Agrawal et al., so hallucination rates can be compared roughly with their L2-7B figure (68.3%), allowing for the different ground-truth source.

---

## 5. Resource access

| Resource | Status | Licence / terms | Label |
|---|---|---|---|
| `selfcheckgpt` package + code | pip installable | MIT | VERIFIED (README) |
| `wiki_bio_gpt3_hallucination` | 238 passages, 20 samples each | CC-BY-SA-3.0 | VERIFIED (dataset card) |
| Agrawal et al. code | github.com/microsoft/hallucinated-references | MIT | VERIFIED (repo page); contents of `code/` and DATA.md **UNKNOWN** |
| Generator: Qwen2.5-7B-Instruct | HF, not gated, 7.61B params | Apache-2.0 | VERIFIED (model card); Colab 4-bit fit **UNKNOWN** until load test |
| Optional anchor generator: Llama-2-7B-Chat | HF, gated by Meta licence acceptance | Llama 2 Community Licence | **UNKNOWN**: needs an access request; approval time unknown |
| NLI checkpoint used by `SelfCheckNLI` | Default checkpoint name and licence | — | **UNKNOWN**: check the source code |
| OpenAlex API | No key: 10 req/s; free key gives 10× daily budget | Data CC0 (from memory) | VERIFIED for rate limits; licence **INFERRED** |
| Crossref REST API | No key; polite pool with `mailto`: 10 req per interval, concurrency 3 | — | VERIFIED (Crossref docs) |
| Bing Web Search API (Agrawal's oracle) | Believed to have been retired by Microsoft in 2025 | — | **INFERRED** from memory. Not needed: OpenAlex/Crossref replace it, and that is a documented deviation. |
| ACM CCS topic list | Publicly browsable | ACM terms | **UNKNOWN**: check whether Agrawal's repo includes the 543 topics |

---

## 6. Compute and time estimate (Colab T4, INFERRED; replaced by measured timings on Day 3)

| Step | Approx. workload | Estimate |
|---|---|---|
| Baseline: SelfCheck-NLI on WikiBio | ~238 passages × ~8 sentences × 20 samples ≈ 38k NLI pairs | 20–60 min |
| Baseline: SelfCheck-Unigram | CPU | < 5 min |
| Generation: 100 topics × (1 greedy + 10 samples) | ~1,100 generations × ~300 tokens, batched | 1–2 h |
| DQ1–3: 500 refs × 3 prompts × 10 samples | Short answers, batched | ~1 h |
| IQ: 500 refs × 3 author queries + 3 pairwise judgements | Short | ~1 h |
| D4 verbalised confidence; D3 log-prob | One pass each | < 30 min |
| SelfCheck-NLI on references | 500 × 10 pairs | < 10 min |
| OpenAlex + Crossref lookups | ~500 refs + 100 topic counts at ≤ 10 req/s | < 15 min |
| Manual audit | 50 references | 2–3 h (student) |

Total GPU time is about 4–6 h, split across several sessions. **Checkpoint results to Google Drive after every batch**, because free Colab sessions can disconnect.

---

## 7. Fallbacks

| If… | Then… |
|---|---|
| Qwen2.5-7B is too slow or does not fit 4-bit on a T4 | Use Qwen2.5-1.5B-Instruct (licence **UNKNOWN**: check before use; Qwen2.5-3B may carry a non-Apache licence) and reduce topics to 60 |
| The `selfcheckgpt` package fails to install | Reimplement NLI scoring directly with the same checkpoint via `transformers` (log as a deviation) |
| The baseline misses the README numbers by more than 2 AUC-PR points and the cause cannot be found | Report the discrepancy and continue with the extension, with the validity threat stated |
| Too few hallucinated (or too few real) references | Add topics. Class balance at L2-7B was ~68% hallucinated, so this is unlikely. |
| OpenAlex misses books or venues | Fall back to Crossref; add "not-found in both" plus manual audit; report the label-noise rate |
| Llama-2 access not granted in time | Drop the anchor model and compare with Agrawal's reported L2-7B numbers only qualitatively |

---

## 8. Go / no-go

**GO**, on these conditions, which must be checked by the end of Day 3:
1. Qwen2.5-7B loads in 4-bit on Colab and the timing test fits the budget above.
2. `SelfCheckNLI` runs on 5 WikiBio passages.
3. The student has read Agrawal et al. (full text) and SelfCheckGPT (results table).
4. OpenAlex returns sensible matches for 10 known-real and 10 invented titles (pilot).

If condition 1 fails, apply the fallback. If conditions 2 and 4 both fail, switch to gap G5 or re-plan.
