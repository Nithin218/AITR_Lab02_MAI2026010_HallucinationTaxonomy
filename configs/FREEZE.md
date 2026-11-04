# Protocol v1 freeze

Frozen: 2026-11-04 (git tag `protocol-v1`)
Signed off by: Nagireddy Nithin Kumar Reddy (MAI2026010). Instruction: "do it without running on colab".

Frozen documents: `docs/07_experiment_protocol.md`, `configs/protocol_v1.yaml`, `configs/{pilot,main,e2}_topics.txt`.
Decisions included: D-001 to D-017 (`research_decisions.md`).

Checklist (`docs/07` §11):
- [x] Primary metric AUROC (y = 1 hallucinated). Primary contrasts: J-NLI vs J-LM (E1), D1 vs D6 (E2)
- [x] Pilot topics in `configs/pilot_topics.txt`; main = the other 190; E2 = 100 of main
- [x] τ = 90, `fuzz.ratio`, name-parsing and refusal rules set on pilot only (D-017)
- [x] `neural_ans{k}` mapping resolved (D-015)
- [x] Empty / unparseable / refusal rules (protocol §3–4, D-017)
- [x] Seeds (§5); versions in `requirements-lock.txt`
- [x] V2 done (36/36, `baseline_validation.md`)
- [ ] V1 done before E1/E2 scoring: runs first in the execution order below

Execution environment (student instruction, 2026-11-04): **local Mac** (Apple M-series GPU via PyTorch MPS, 8 GB), in a project venv with the same package versions as the Colab lock file (torch 2.11.0 without CUDA). NLI label order was re-checked on MPS: 0.0002 / 0.9997, identical to Colab.
Execution order: V1 → E1 → R1 (R1 is network-bound and may run alongside).

Any change after this point is logged as PC-xxx in `research_decisions.md`.
