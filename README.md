# Do Zero-Resource Detectors Catch Fabricated References in Small LLMs?

AITR Lab 02: implementation study following the T20 review *A Taxonomy of Hallucination Types in Large Language Models for Research Applications*.
Nagireddy Nithin Kumar Reddy (MAI2026010), IIIT Allahabad.

**Question.** Agrawal et al. (2023) found that LM-judged consistency checks detect hallucinated references for GPT-4 but not for 7B–13B Llama-2 models. Does replacing the LM judge with an external NLI judge (as in SelfCheckGPT) recover detection at small scale? How does detection depend on topic familiarity? Details: `docs/05_research_brief.md`, protocol `docs/07_experiment_protocol.md`.

Status: **setup. No experimental results yet.**

## Layout
```
README.md                 this file
requirements.txt          proposed dependency ranges (exact versions: requirements-lock.txt)
configs/                  protocol_v1.yaml, pilot/main topic lists
data/README.md            data sources, licences, download procedure
docs/                     01–07: scope, gaps, mapping, novelty, brief, data sheet, protocol
notebooks/                00_environment_check.ipynb (Colab)
src/aitr/                 shared code (seeding, downloads; detectors added in Prompt 09)
scripts/                  00_env_check.py, 01_make_pilot_split.py
tests/                    unit/sanity tests
results/                  genuine outputs only (env/, raw/, metrics, figures)
paper/                    LaTeX source (later)
research_decisions.md     decision log (D-xxx) and protocol changes (PC-xxx)
baseline_validation.md    V1/V2 reproduction checks
env_verification_log.md   environment test outputs
ai_assistance_log.md      AI-use record
```

## Reproduce the setup (Colab, T4)
Open `notebooks/00_environment_check.ipynb` and run all cells. Commands executed so far are logged in `env_verification_log.md`.
