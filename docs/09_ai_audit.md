# AI-Use and Research-Integrity Report (Prompt 18)

Project: AITR Lab 02 (MAI2026010). Period: 2026-10-29 to 2026-11-04.
AI tool: Claude Code (model Claude Opus 5.5), used as a pair programmer and research assistant with shell access to the student's Mac. Student tools: Google Colab (T4), Overleaf, and this repository.
Sources for this report: `ai_assistance_log.md` (entries 1–18), `research_decisions.md` (D-001 to D-019, PC-001 to PC-005), and the git history (`git log`). Nothing below is reconstructed from memory.

## 1. Significant AI-assisted activities
| Task | AI proposal | Human decision | Verification | Outcome |
|---|---|---|---|---|
| Gap extraction (P02) | 5 gaps; ranked G1 first | Accepted | Later novelty search | G1 later found to be already demonstrated (row 3) |
| Novelty search (P04) | Found Agrawal et al. 2023; narrowed the question | Accepted the narrowed RQ | arXiv full text read by fetch; data downloaded | Study redesigned (D-007) |
| Study design (P06–07) | Judge-swap on released data (E1) | Accepted; approved E2 on 100 topics and 13B dedup | Data schema inspected; label rates matched the paper | Protocol v1 |
| Environment (P08) | Env check script | Ran it on Colab | Real Colab output (9/10 PASS) | `pip_check` fixed (D-012) |
| Baselines (P09–10) | Loader, V2 reproduction | — | Tolerance committed before computing; 36/36 | Data defect found (D-014) |
| Execution (P09–14) | Local MPS execution; batched NLI | Student: no Colab; drop full V1-NLI; drop E2 | Equivalence 2e-7; V1 Unigram 85.64 vs 85.63 | PC-001 to PC-004 |
| Writing (P15–16) | Draft, figures from CSVs, self-review | _[student to review]_ | Each number cross-checked against the CSVs | Revisions (docs/08) |

## 2. Real examples of AI limitations, corrections and purposeful verification
1. **Overclaimed novelty.** The AI's first gap ranking (P02–03) presented "can self-consistency detect fabricated references?" as open. Its own P04 search found Agrawal et al. had already shown this. The framing would have overclaimed novelty. *Corrected in D-007.*
2. **Assumption about released data.** The loader test assumed every file had 200 topics × 5 references. Running it against the real file failed: Llama-2-13B repeats 200 references and lacks 40 topics. *A data defect upstream, found by verification (D-014).*
3. **Auto-flip in the original AUC procedure.** Reading roc-utils' source showed that Agrawal's AUCs silently invert below-chance predictors. Copying the method into our own experiments would have hidden inverted judges. *Restricted to V2; tested (D-016).*
4. **Name parser broke on real answers.** The first J-STR parser, written before seeing the data, failed on three real answer formats. *Caught on the pilot before any main result (D-017).*
5. **Refusals scored as agreement.** NLI makes two refusals look consistent. The AI noticed this only after inspecting unparsed pilot answers. *Rule aligned with Agrawal's judge instruction; the raw variant kept as a sensitivity check.*
6. **False match from an order-insensitive similarity.** The protocol's `token_sort_ratio` matched a reordered fake title. *Replaced on pilot evidence (κ 0.67 → 0.80).*
7. **Runtime bug.** E1 crashed with a `KeyError` after scoring GPT-4, caused by a merge onto placeholder columns. *Fixed from the actual traceback, with a regression test and resume from checkpoint (D-018).*
8. **Compute misestimates.** E2 was estimated at 4–6 GPU-h. It measured 8.6 GPU-h on a T4 and 334 s/topic on the Mac. *This led to scope changes, logged with measured evidence (D-013, PC-004).*
9. **Unsupported claims in the draft.** The first paper draft contained an unverified parameter count, an inaccurate runtime statement, a "Bonferroni-style" claim that was never computed, and an abstract claim about an analysis not yet run. *All removed before commit (AI log 18; docs/08).*

## 3. Conclusions that depended on human reasoning and decisions
- The scope decisions: running locally instead of on Colab, dropping V1-NLI re-benchmarking (PC-002), switching the generator and then dropping E2 (PC-003/004), adding RQ2 on released data (PC-005).
- Sharing an OpenAlex key and an institutional email with external APIs (D-019).
- Accepting the narrowed research question after the novelty search.
- **Still required from the student:** read and own the code (especially `src/aitr/judges.py`, `metrics.py`, `agrawal.py`); confirm or replace the decision rules (±2.0 AUC-PR, ±0.005, τ = 90); review the paper's interpretation, in particular the discussion's mechanism claim. The "human decision" column of `ai_assistance_log.md` still contains placeholders that only the student can fill honestly.

## 4. What was learned about vibe coding here
- Running code against the real data found more problems than reading it did: items 2, 4 and 7 were invisible until execution.
- Committing targets before computing (V1, V2, RQ2) kept the AI from fitting the analysis to the results, and the git timestamps make that checkable.
- An AI is fast at producing plausible-looking pipelines. The expensive part is the checks: equivalence tests, pilot inspection, cross-checking every number.

## 5. Remaining risks
- Most code was written by the AI. The student must be able to explain it in the viva.
- Several runs were executed by the AI on the student's machine, not by the student. They are logged and reproducible, but the student should rerun at least V2 and E1 from the cache (README checklist).
- The SelfCheck-NLI benchmark number was not re-measured. J-STR's parser was tuned on pilot data from the same models.
- External API results (OpenAlex, Crossref) can change over time. The cached responses in `results/raw/oracle_cache.jsonl` are the record.
