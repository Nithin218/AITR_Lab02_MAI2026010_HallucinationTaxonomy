# Technical Peer Review of the Draft (Prompt 16)

Draft reviewed: `paper/main.tex` at commit `2832991` (2026-11-04). Reviewer stance: critical. Praise is omitted.

## Summary of the submission
The paper reproduces Agrawal et al.'s reference-hallucination consistency checks from released data, then swaps the consistency judge (LM vs NLI vs string overlap) while holding the sampled answers fixed. It concludes that small-model failure lies in the answers, not the judge.

## MAJOR concerns

**M1. H1's scope is presented more narrowly than it was registered (§3, §5.2).** The frozen protocol (§6) defines the E1 contrast J-NLI − J-LM for *every* model, and the brief's H1 targets "the 7B setting". The draft states H1 as "on the small model" and treats the other five models as context. That reads as a post-hoc choice of the favourable model and hides six tests.
*Why it matters:* readers cannot tell which result was decisive or how much the risk of a false positive grows across six tests.
*Remedy:* state that the contrast was registered for all six models, that H1 was motivated by the small models, and that results are reported for all six without correction for multiple tests. Name Llama-2-7B as the model closest to the brief's 7B target.

**M2. Label noise could explain near-chance AUROC at 7B (§5.2, §6).** Bing exact-phrase search "grounds" generic phrases that are not paper titles. The pilot (D-017) found 8 of 20 Bing-grounded Llama-2-7B titles only partially matched in scholarly databases. Llama-2-7B produces the most generic titles. If its labels are noisier, every judge's AUROC is pulled towards 0.5, and "the signal is missing from the answers" would not follow.
*Why it matters:* this is a direct alternative explanation of the headline claim.
*Remedy (cheap):* recompute E1 AUROC for Llama-2-7B on the subset where Bing and the OpenAlex/Crossref oracle agree (R1 labels). If the judges stay near chance there, label noise is not the explanation. Report it as a **post-hoc** robustness analysis.

**M3. The mechanism is stated too strongly (§6, abstract, conclusion).** "The failure lies in what the small model knows about authorship" is an inference. Alternatives: three samples are too few to estimate consistency; temperature or prompt settings in the released data; a shared failure in the answer format, for example Llama-2's chat boilerplate.
*Remedy:* soften to "is consistent with", list the alternatives, and move the claim out of the abstract's final sentence.

## MINOR concerns
- **m1 (§4.2).** J-STR's parser was tuned on pilot answers from the same six models, so its advantage could partly reflect that tuning. State it, and note that the pilot topics are excluded from all results.
- **m2 (§5.3).** RQ2's familiarity proxy (leaf-phrase search count) can be dominated by meanings outside computing. Already in the limitations; add one example of an extreme count to the results.
- **m3 (§1).** The author's own earlier review is mentioned but not cited. Add it as an unpublished manuscript.
- **m4 (Table 2).** Six columns with CIs may overflow the `acmsmall` text width. Check the compiled PDF.
- **m5 (§2).** The URL-level citation audit (Rao et al., arXiv:2604.03173, metadata checked 2026-10-29) is relevant neighbouring work and could be cited.
- **m6 (§4.5/§5.1).** Make explicit that V2's auto-flip was never triggered, so V2 and E1 numbers are directly comparable.

## Prioritised revision checklist
| # | Concern | Action | Status |
|---|---|---|---|
| 1 | M2 | Post-hoc E1 on the Bing–oracle agreement subset (Llama-2-7B) after R1 | pending R1 |
| 2 | M1 | Rewrite §3 and §5.2: all six registered, multiplicity stated | done in revision |
| 3 | M3 | Soften the mechanism claim; add alternatives | done in revision |
| 4 | m1, m3, m5, m6 | Text and bib additions | done in revision |
| 5 | m4 | Check the table width in the compiled PDF | after compile |
| 6 | m2 | Add the extreme-count example | done in revision |

## Author-response table
| Concern | Response | Location in revision |
|---|---|---|
| M1 | Agreed. H1 is restated for all six registered contrasts, with no multiplicity correction and Llama-2-7B named as the brief's target | §3, §5.2 |
| M2 | Agreed. A post-hoc agreement-subset analysis is added and labelled as not pre-registered | §5.3 |
| M3 | Agreed. The claim is softened to "consistent with" and three alternatives are listed | §6, abstract, conclusion |
| m1 | Agreed; stated in §4.2 and the limitations | §4.2, §7 |
| m2 | Agreed; the largest count ("Topology"-type leaf phrases) is reported | §5.4 |
| m3 | Agreed; cited as an unpublished manuscript | §1, bib |
| m4 | Checked after compiling | Table 2 |
| m5 | Added | §2, bib |
| m6 | Added | §5.1 |
