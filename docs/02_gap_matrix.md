# Gap Matrix (Prompt 02 output)

Source: `../../main.tex` (line numbers cited). Labels: VERIFIED / INFERRED / UNKNOWN.

Key distinction in every row:
- **(O)** a limitation reported *in the original paper*. This needs checking against that paper; the review only paraphrases it.
- **(R)** a limitation or direction stated *by the review author*.
- **(I)** *my inference*, not stated anywhere yet.

No gap below has been through a novelty search. Novelty status is **UNKNOWN** for all of them (Prompt 04).

---

## 1. Central elements extracted from the review

**Methods:** SelfCheckGPT, FActScore, semantic entropy, P(True)/self-knowledge (Kadavath), FAVA, RAG, bibliographic audits (manual or database-backed).
**Datasets/benchmarks:** HaluEval, Med-HALT, RAGTruth; the citation-audit samples of Walters & Wilder, Chelli et al., Linardon et al. and others.
**Measured findings:** reported only qualitatively in the review (see `01_review_scope.md` §5).
**Unresolved questions in the review:** existence vs. support (l.197, 282); self-assessment (l.296); long tail (l.293); topic dependence (l.302); benchmark coverage (l.305); faithfulness under RAG (l.299).

---

## 2. Five computationally testable gaps

### G1. Can cheap zero-resource signals detect fabricated citations as well as a database lookup?
- **Review evidence:** l.197: "citation verification requires checking at least two properties: whether the cited work exists and whether the cited work supports the associated claim… motivates integrated verification pipelines that query authoritative bibliographic sources". Also l.282 and future direction 2 (l.312).
- **Source type:** (R) the pipeline proposal is the review's. (O) Walters & Wilder, Bhattacharyya et al. and Chelli et al. checked references manually or against databases (review l.197). *How* each paper verified its references needs checking against the original.
- **Inference (I):** no reviewed paper compares a database check against model-internal signals (self-consistency, verbalised confidence) on the same set of generated references.
- **Experimental question:** On LLM-generated references, how well do (a) verbalised confidence and (b) SelfCheckGPT-style consistency across samples detect nonexistent references, compared with (c) a metadata lookup in Crossref/OpenAlex as ground truth?
- **Evidence that would test it:** a labelled set of generated references (exists / metadata mismatch / fabricated); AUROC, precision and recall per detector; cost per reference.
- **Taxonomy link:** Axis C (citation fabrication), Axis A (calibration).

### G2. Does citation fabrication increase as topics become less familiar (long tail)?
- **Review evidence:** l.302: "Linardon et al. report topic-dependent differences in citation fabrication". l.293: "calibrated language models can still hallucinate on rare facts… a newly published or obscure paper may have weak representation in training data".
- **Source type:** (O) Linardon et al.'s topic-familiarity finding, which needs checking: the topics, the models, and how "familiarity" was operationalised. (O) Kalai & Vempala's theoretical claim, which needs checking: the exact assumptions. (R) the link to citation generation is the review's (l.293).
- **Inference (I):** familiarity can be measured objectively, e.g. by the publication count for a topic in OpenAlex, instead of an expert's judgement. Linardon's domain was mental health (review l.273); whether the effect holds outside it is untested in the review's evidence.
- **Experimental question:** Across topics stratified by publication volume, does the fabrication rate rise as volume falls, and do detectors from G1 lose accuracy on rare topics?
- **Evidence that would test it:** fabrication rate per stratum, with confidence intervals and a trend test; detector AUROC per stratum.
- **Taxonomy link:** Axis A (long tail), Axis C.

### G3. Is an LLM's verbalised confidence about its own citations calibrated?
- **Review evidence:** l.296: "Dahl et al. show that models may fail to recognize legal hallucinations, and this limits self-verification". l.137: Kadavath et al. "found evidence that confidence can carry information about correctness under some settings".
- **Source type:** (O) Dahl et al.'s finding (legal domain) and Kadavath et al.'s (general QA). The exact settings need checking against both originals. (I) whether either result carries over to bibliographic references.
- **Experimental question:** When asked "how confident are you that this reference exists?", is the model's stated probability calibrated (ECE, reliability diagram) against database ground truth?
- **Evidence that would test it:** confidence scores alongside ground truth; ECE, Brier score, AUROC.
- **Note:** overlaps heavily with G1. It is best treated as detector (a) inside G1 rather than as a separate study.

### G4. Does SelfCheckGPT keep its detection performance when moved from its original domain to research-paper claims?
- **Review evidence:** l.185: "repeated sampling introduces computational cost and does not directly establish the truth of a stable but incorrect statement". l.191: general benchmarks "do not necessarily cover research-native errors".
- **Source type:** (O) the stable-but-wrong limitation is plausibly in the SelfCheckGPT paper itself (needs checking). (R) the benchmark-coverage claim is the review's. (I) the original evaluation domain and release of code and data are **UNKNOWN**. From memory it is GPT-3-generated WikiBio passages with a public package and dataset, but this is unchecked and must be verified.
- **Experimental question:** After SelfCheckGPT is reproduced on its original benchmark, how far does its AUC-PR drop on sentences describing research papers (e.g. LLM-written abstract summaries checked against the real abstract)? What fraction of hallucinations stay consistent across samples ("stable errors")?
- **Evidence that would test it:** reproduced baseline numbers within tolerance of the published ones; AUC-PR on the new domain; the fraction of stable errors.
- **Pathway:** A (reproduction + extension). Risk: building research-domain labels takes manual annotation.

### G5. Under RAG, which detector best catches answers that are unfaithful to the retrieved evidence?
- **Review evidence:** l.221: "hallucination can persist even when relevant retrieved passages are supplied". l.299: "A model can retrieve the correct paragraph and still summarize it incorrectly… motivates claim-level provenance".
- **Source type:** (O) RAGTruth's finding (needs checking: its tasks, its annotation, and its own baseline detectors). (R) the claim-level provenance direction (l.313).
- **Experimental question:** On RAGTruth, how do an NLI-entailment detector, an LLM-as-judge and SelfCheckGPT compare for response-level hallucination detection, and does performance differ by task type?
- **Evidence that would test it:** precision, recall and F1 per detector per task, against RAGTruth labels.
- **Risk:** dataset access and licence are **UNKNOWN**. NLI and LLM-judge inference over thousands of responses may need a GPU or API budget.

---

## 3. Ranking (top three)

Scored 1–3 (3 = best) on significance, resource needs (with the student's hardware still UNKNOWN), and implementation feasibility.

| Rank | Gap | Significance | Resources | Feasibility | Total | Rationale |
|---|---|---|---|---|---|---|
| 1 | **G1** (with G3 inside it) | 3 | 3 | 3 | **9** | Matches the review's central distinction (l.197, 282). Builds its own ground truth from public databases. Runs on a laptop. Has natural comparators: database lookup, self-consistency, verbalised confidence. |
| 2 | **G2** | 3 | 3 | 2 | **8** | Same pipeline as G1, so it serves as G1's deeper analysis (stratification and robustness). Choosing the familiarity metric needs justification. |
| 3 | **G4** | 2 | 2 | 2 | **6** | A true Pathway A reproduction, which strongly satisfies G2 ("verified baseline"). The research-domain labels need manual annotation, and code/data access is UNKNOWN. |
| — | G5 | 3 | 1 | 1 | 5 | Significance is high, but access and compute are UNKNOWN. Keep as a fallback. |

**Provisional recommendation (INFERRED, pending N1–N3 in `research_decisions.md`):** a single study combining G1 (primary question, Pathway B controlled comparison) and G2 (deeper analysis). Keep G4 as a fallback if a reproduced published baseline turns out to be required by the grader. The assignment asks for a "verified baseline" (G2 gate); for G1 the baseline would be a reimplemented SelfCheckGPT detector, checked against its paper's description.

---

## 4. Claims to verify against the original publications

| # | Claim as written in the review | Line | What to check in the original |
|---|---|---|---|
| V1 | Walters & Wilder "documented substantial fabrication and error rates" | 77 | Actual rates; models (versions); sample size; verification method |
| V2 | Chelli et al. show "persistent reference-accuracy problems" for systematic reviews | 77, 267 | Rates per system; how references were checked |
| V3 | Linardon et al. report "topic-dependent" fabrication | 151, 273, 302 | How familiarity was defined; effect sizes; model |
| V4 | Kalai & Vempala: calibrated models "can still hallucinate on rare facts" | 137, 293 | Exact theorem and assumptions; whether "must" or "can" is accurate |
| V5 | Dahl et al.: models "fail to recognize legal hallucinations" | 296 | What was measured (self-awareness test?); which models |
| V6 | Kadavath et al.: confidence "can carry information about correctness under some settings" | 137 | Which settings; which model sizes; P(True) method |
| V7 | SelfCheckGPT works without an external knowledge base; stable-but-wrong limitation | 185 | Whether the paper states this limitation; dataset; released code |
| V8 | RAGTruth: hallucination persists with relevant passages | 221, 299 | Tasks, annotation scheme, data licence, baseline detectors |
| V9 | Magesh et al.: legal RAG tools "reduce but not eliminate" hallucination | 159, 221 | Tools evaluated; rates; whether "reduce" is relative to GPT-4 |
| V10 | Fabricated references reported in "legal" settings via the cited Cureus/Sci. Rep. papers | 61 | Likely citation-scope mismatch (see `01_review_scope.md` §7) |

## 5. What this matrix does **not** establish
- That any gap is novel (no search done)
- That any dataset, code repository or API is accessible, or under what licence
- Any numeric result from any paper
