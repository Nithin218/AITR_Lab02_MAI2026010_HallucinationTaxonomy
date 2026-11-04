# Five-Minute Demonstration Script (submission item E)

Setup before the demo: open a terminal in the repo with `.venv` ready, open the paper PDF, and have `research_decisions.md` open in an editor. Everything below runs offline from cached data in under 2 minutes in total.

| Time | Show | Say (key points) | Command / file |
|---|---|---|---|
| 0:00–0:40 | Paper title and Figure 1 | Agrawal et al.: consistency checks catch fake references for GPT-4 but fail for 7B models. My question: is the failure in the *judge* or in the *answers*? | `paper/main.pdf` |
| 0:40–1:30 | Baseline verified | Before claiming anything, I reproduced all 36 of their published AUCs from released scores, with the tolerance fixed in advance. | `python scripts/10_v2_reproduce_agrawal.py` → last line `V2: 36/36 cells within ±0.005` |
| 1:30–2:00 | Data defect | The 13B file repeats 200 references. The tests caught it. | `python tests/test_agrawal.py` |
| 2:00–3:15 | Main result | The same 3 author answers, three judges. On 7B all are near chance; NLI is worse elsewhere; plain name overlap is as good as the LM. | `python scripts/12_e1_judge_swap.py --split main --judges str` (recomputes from cache) · `results/e1_deltas.csv` |
| 3:15–3:45 | One judge in code | How J-STR parses names; the test cases come from real pilot answers. | `src/aitr/judges.py` · `python tests/test_judges.py` |
| 3:45–4:20 | Integrity | Protocol frozen before results (`git show protocol-v1`). Five logged changes, each saying whether results had been seen. | `git log --oneline --decorate | grep protocol-v1` · `research_decisions.md` (PC-001..005) |
| 4:20–5:00 | Limits and AI use | Label-noise check; no new generator run; NLI benchmark not re-measured; AI log with real corrections. | `docs/09_ai_audit.md` |

## Likely viva questions and where the answer is
- *Why is AUROC computed with a topic bootstrap?* Five references share a topic; `src/aitr/metrics.py::cluster_bootstrap_auc`.
- *Why not auto-flip like Agrawal?* It would hide a judge that works backwards; D-016 and `tests/test_metrics.py::test_perfect_and_inverted`.
- *What does the refusal rule do, and why is it fair?* Agrawal's LM judge was given the same instruction; D-017 #2; the J-NLI-raw column.
- *Why was E2 dropped?* Measured 334 s per topic; PC-004.
- *Is your NLI scorer the same as SelfCheckGPT's?* Equivalence check, max difference 2e-7; `baseline_validation.md`.
- *Rerun one component live:* `python scripts/10_v2_reproduce_agrawal.py` (about 1 min, CPU).
