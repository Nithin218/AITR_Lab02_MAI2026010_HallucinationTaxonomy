# Review Scope Summary (Prompt 01 output)

Source read: `../../main.tex` (335 lines) and `../../references.bib` (24 entries). Line numbers below refer to `main.tex` as it stood on 28 Sep 2026.
Not read: `MAI2026010_Review_Paper_FINAL.pdf`, `Figures.pdf`, `Reflection.pdf`. Their contents are **UNKNOWN**.

## 1. What the review is

| Item | Content | Label |
|---|---|---|
| Title | A Taxonomy of Hallucination Types in Large Language Models for Research Applications | VERIFIED (l.30) |
| Type | Narrative literature review and conceptual synthesis. No experiments of its own. | VERIFIED (whole file) |
| Format | ACM `acmsmall`; 2 TikZ figures; 3 tables | VERIFIED |
| Domain | LLM use in scholarly workflows: literature synthesis, citation retrieval, drafting, interpretation | VERIFIED (l.44, l.57) |

## 2. Main contribution: three-axis taxonomy (l.79–179)

H = H_A × H_B × H_C, a conceptual product in which one error can carry several labels (l.81–84).

- **Axis A, foundational:** intrinsic/extrinsic (Ji et al.); factuality/faithfulness (Maynez et al., Huang et al.); calibration / long tail (Kadavath, Kuhn, Kalai & Vempala).
- **Axis B, content level** (following FAVA): entity, relational, invented, contradictory, unverifiable/overclaimed, incomplete/omissive.
- **Axis C, research specific:** citation/reference; empirical-finding misattribution; numerical/statistical; methodological/reasoning; legal/medical/professional.
- Table 1 (l.163–179) maps six example failures onto all three axes.

## 3. Methods surveyed

| Family | Representative work in review | Review's stated limitation | Label |
|---|---|---|---|
| Consistency | SelfCheckGPT (l.185) | Sampling is costly; stable output does not mean true | VERIFIED as review text |
| Atomic decomposition | FActScore (l.188) | Needs a knowledge source and a definition of "supported" | VERIFIED as review text |
| Benchmarks | HaluEval, Med-HALT (l.191) | Do not cover research-native errors | VERIFIED as review text |
| Uncertainty | Kadavath; semantic entropy (l.194) | Uncertainty ≠ hallucination | VERIFIED as review text |
| Citation verification | Walters & Wilder, Bhattacharyya, Chelli (l.197) | Manual or database checks; two properties: existence and support | VERIFIED as review text |
| Mitigation | RAG (Lewis, Gao), RAGTruth, FAVA, calibration/abstention (l.216–227) | RAG reduces hallucination but does not remove it | VERIFIED as review text |

## 4. Datasets and benchmarks named

HaluEval, Med-HALT, RAGTruth, plus the citation-audit samples of the individual studies. The review names **no dataset sizes, licences or access routes**, so all access details are **UNKNOWN**.

## 5. Quantitative findings in the review

**None.** The review describes results only qualitatively ("substantial fabrication and error rates", l.77; "substantial error rates", l.159; "reduce but not eliminate", l.159/221). Any number used in the new paper must come from the original publications (**UNKNOWN** until read).

## 6. Future directions the review itself states (l.310–318)

1. Research-native benchmarks
2. Automated bibliographic verification
3. Claim-level provenance
4. Calibrated abstention
5. Cross-domain / cross-lingual validation
6. Human-in-the-loop verification
7. Longitudinal auditing of propagation

These are the review author's own conclusions. They are **not** limitations reported by the original papers (see the gap matrix for that distinction).

## 7. Observations about the review itself (for the student to check)

- l.61 says fabricated references are documented in "scientific, medical, **legal**, geographic, and systematic-review settings" and cites Athaluri, Bhattacharyya, Day, Walters and Chelli. None of those five is a legal study; legal evidence comes from Dahl et al. and Magesh et al., which are cited separately. **INFERRED** citation-scope mismatch.
- `alkaissi2023artificial` is in `references.bib` but is never cited in `main.tex`. **VERIFIED** (grep count = 0).
- The review lists 23 cited works. They are the evidence base for this project, but each claim must be re-checked against the original paper (assignment §2, "Use that review as your evidence base, not as text to recycle").
