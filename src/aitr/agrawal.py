"""Load the six released Agrawal et al. result files into one schema (protocol §3).

Unified columns (one row = one generated reference):
    model        e.g. "llama-2-7b-chat"
    row          row index inside the original file (0..999)
    topic        ACM CCS topic the reference was generated for
    gen_title    the generated title
    main_answer  the model's full answer listing 5 titles
    y            1 = hallucinated (bing_return == "False"), 0 = grounded
    iq_1..iq_3   the 3 independently sampled author answers (as released, "" if empty)
    iq_lm        released LM-judged author agreement in [0, 1] (higher = more consistent)
    dq1..dq3     released fraction of "yes" answers in [0, 1] (higher = "exists")

Column mapping, verified from upstream code at commit bf3ef5f:
  GPT files:   code/metrics.ipynb uses neural_ans1_prob as IQ and neural_ans2/3/4_prob as DQ1/2/3.
  Llama files: code/src/llama_run.py writes IQ_llama_prob (mean pairwise agreement) and
               DQ{k}_prob_sample (fraction of answers containing "yes").
Released scores are NOT re-oriented here; detectors flip them later (s = 1 - value).
"""
from pathlib import Path

import pandas as pd

GPT_MODELS = ["gpt-4", "gpt-3.5-turbo", "text-davinci-003"]
LLAMA_MODELS = ["llama-2-7b-chat", "llama-2-13b-chat", "llama-2-70b-chat"]
MODELS = GPT_MODELS + LLAMA_MODELS

_GPT_MAP = {"title": "topic", "model_answer_main_query": "main_answer",
            "model_ans_1": "iq_1", "model_ans_2": "iq_2", "model_ans_3": "iq_3",
            "neural_ans1_prob": "iq_lm", "neural_ans2_prob": "dq1",
            "neural_ans3_prob": "dq2", "neural_ans4_prob": "dq3"}
_LLAMA_MAP = {"concept": "topic", "model_answer_main_query": "main_answer",
              "IQ_ans1": "iq_1", "IQ_ans2": "iq_2", "IQ_ans3": "iq_3",
              "IQ_llama_prob": "iq_lm", "DQ1_prob_sample": "dq1",
              "DQ2_prob_sample": "dq2", "DQ3_prob_sample": "dq3"}
UNIFIED = ["model", "row", "topic", "gen_title", "main_answer", "y",
           "iq_1", "iq_2", "iq_3", "iq_lm", "dq1", "dq2", "dq3"]
TEXT_COLS = ["topic", "gen_title", "main_answer", "iq_1", "iq_2", "iq_3"]
SCORE_COLS = ["iq_lm", "dq1", "dq2", "dq3"]


def file_name(model: str) -> str:
    return f"{model}_results.csv" if model in GPT_MODELS else f"{model}.csv"


def load_model(raw_dir: Path, model: str) -> pd.DataFrame:
    """Read one released file and return it in the unified schema. Raises on unexpected values."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    # dtype=str + keep_default_na=False: nothing is silently coerced; empty cells stay "".
    df = pd.read_csv(Path(raw_dir) / file_name(model), dtype=str, keep_default_na=False)
    mapping = _GPT_MAP if model in GPT_MODELS else _LLAMA_MAP
    missing = [c for c in ["gen_title", "bing_return", *mapping] if c not in df.columns]
    if missing:
        raise KeyError(f"{model}: columns missing {missing}")

    labels = set(df["bing_return"].unique())
    if not labels <= {"True", "False"}:
        raise ValueError(f"{model}: unexpected bing_return values {labels - {'True', 'False'}}")

    out = df.rename(columns=mapping)
    out["model"] = model
    out["row"] = range(len(out))
    out["y"] = (df["bing_return"] == "False").astype(int)
    for c in SCORE_COLS:
        out[c] = pd.to_numeric(out[c], errors="raise")
    for c in TEXT_COLS:
        out[c] = out[c].str.strip()
    return out[UNIFIED]


def load_all(raw_dir: Path, models=MODELS) -> pd.DataFrame:
    return pd.concat([load_model(raw_dir, m) for m in models], ignore_index=True)


def drop_duplicate_refs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first copy of each (model, topic, gen_title).

    Needed for llama-2-13b-chat: rows 700-899 repeat 200 (topic, title) pairs from rows 500-699
    with identical bing labels but re-sampled IQ/DQ answers, while 40 topics are absent
    (found 2026-10-30, research_decisions.md D-014). Other files have no duplicates.
    """
    return df.drop_duplicates(["model", "topic", "gen_title"], keep="first").reset_index(drop=True)


def validate(df: pd.DataFrame, topics: list[str]) -> dict:
    """Integrity checks for one model's frame. Returns a report; raises only on hard violations."""
    if len(df) != 1000:
        raise ValueError(f"expected 1000 rows, got {len(df)}")
    unknown = set(df["topic"]) - set(topics)
    if unknown:
        raise ValueError(f"{len(unknown)} topics not in acm_ccs_200, e.g. {sorted(unknown)[:3]}")
    out_of_range = {c: int(((df[c] < 0) | (df[c] > 1)).sum()) for c in SCORE_COLS}
    if any(out_of_range.values()):
        raise ValueError(f"scores outside [0, 1]: {out_of_range}")
    refs_per_topic = df.groupby("topic").size()
    return {
        "rows": len(df),
        "topics": int(df["topic"].nunique()),
        "refs_per_topic": sorted(refs_per_topic.unique().tolist()),
        "n_hallucinated": int(df["y"].sum()),
        "score_nan": {c: int(df[c].isna().sum()) for c in SCORE_COLS},
        "empty_iq": {c: int((df[c] == "").sum()) for c in ["iq_1", "iq_2", "iq_3"]},
        "empty_gen_title": int((df["gen_title"] == "").sum()),
        "duplicate_titles_within_topic": int(df.duplicated(["topic", "gen_title"]).sum()),
    }
