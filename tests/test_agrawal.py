"""Tests for src/aitr/agrawal.py. Run: python tests/test_agrawal.py (needs data/raw/agrawal; see data/README.md)."""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr import agrawal  # noqa: E402

RAW = ROOT / "data/raw/agrawal"
# Hallucination counts implied by the paper's reported rates (and seen in env runs 0 and 1).
EXPECTED_HALLUCINATED = {"gpt-4": 468, "gpt-3.5-turbo": 596, "text-davinci-003": 736,
                         "llama-2-7b-chat": 683, "llama-2-13b-chat": 767, "llama-2-70b-chat": 662}


def test_label_orientation_on_synthetic_file(tmp_dir=Path(tempfile.mkdtemp())):
    """y must be 1 exactly when bing_return == "False" (getting this wrong flips every AUROC)."""
    pd.DataFrame({
        "concept": ["T", "T"], "gen_title": ["real one", "fake one"], "model_answer_main_query": ["a", "b"],
        "bing_return": ["True", "False"], "IQ_ans1": ["x", ""], "IQ_ans2": ["x", "y"], "IQ_ans3": ["x", "z"],
        "IQ_llama_prob": ["0.9", "0.1"], "DQ1_prob_sample": ["1.0", "0.2"],
        "DQ2_prob_sample": ["1", "0"], "DQ3_prob_sample": ["0.5", "0.5"],
    }).to_csv(tmp_dir / "llama-2-7b-chat.csv", index=False)
    df = agrawal.load_model(tmp_dir, "llama-2-7b-chat")
    assert df["y"].tolist() == [0, 1]
    assert df["iq_1"].tolist() == ["x", ""]          # empty stays "", not NaN
    assert df["iq_lm"].tolist() == [0.9, 0.1]
    assert list(df.columns) == agrawal.UNIFIED


def test_unexpected_label_value_raises(tmp_dir=Path(tempfile.mkdtemp())):
    pd.DataFrame({c: ["1"] for c in ["concept", "gen_title", "model_answer_main_query", "IQ_ans1", "IQ_ans2",
                                     "IQ_ans3", "IQ_llama_prob", "DQ1_prob_sample", "DQ2_prob_sample",
                                     "DQ3_prob_sample"]} | {"bing_return": ["maybe"]}
                 ).to_csv(tmp_dir / "llama-2-7b-chat.csv", index=False)
    try:
        agrawal.load_model(tmp_dir, "llama-2-7b-chat")
    except ValueError:
        return
    raise AssertionError("bing_return='maybe' should raise")


def test_released_files():
    topics = [t.strip() for t in (RAW / "acm_ccs_200.titles").read_text(encoding="utf-8").splitlines() if t.strip()]
    reports = {}
    for model in agrawal.MODELS:
        df = agrawal.load_model(RAW, model)
        rep = agrawal.validate(df, topics)
        assert rep["n_hallucinated"] == EXPECTED_HALLUCINATED[model], (model, rep["n_hallucinated"])
        if model == "llama-2-13b-chat":
            # upstream duplication (D-014): 160 topics, 200 repeated (topic, title) rows
            assert rep["topics"] == 160 and rep["duplicate_titles_within_topic"] == 200, rep
            dedup = agrawal.drop_duplicate_refs(df)
            assert len(dedup) == 800 and dedup["topic"].nunique() == 160
            rep["after_dedup"] = {"rows": len(dedup), "n_hallucinated": int(dedup["y"].sum())}
        else:
            assert rep["topics"] == 200 and rep["refs_per_topic"] == [5], (model, rep)
            assert rep["duplicate_titles_within_topic"] == 0, (model, rep)
        reports[model] = rep
    print(json.dumps(reports, indent=1))


def test_duplicate_copies_share_labels():
    """The repeated 13B rows must carry the same bing label, otherwise dedup would change ground truth."""
    df = agrawal.load_model(RAW, "llama-2-13b-chat")
    groups = df[df.duplicated(["topic", "gen_title"], keep=False)].groupby(["topic", "gen_title"])["y"]
    assert (groups.nunique() == 1).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
