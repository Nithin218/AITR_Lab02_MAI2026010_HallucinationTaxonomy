"""Tests for src/aitr/judges.py and the cluster bootstrap. Run: python tests/test_judges.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr import judges  # noqa: E402
from aitr.metrics import cluster_bootstrap_auc, paired_delta  # noqa: E402


def test_name_tokens_from_real_formats():
    # every input below is copied from a pilot-topic row of the released files (D-017)
    assert judges.name_tokens("Richard S. Sutton, Andrew G. Barto") == {"richard", "sutton", "andrew", "barto"}
    same_line = 'Sure! Here are the authors of the reference "provided reference": Trevor Hastie, Robert Tibshirani'
    assert judges.name_tokens(same_line) == {"trevor", "hastie", "robert", "tibshirani"}
    bullets = 'Sure, I\'d be happy to help! The authors of the reference "provided reference" are: * Richard S. Sutton * Andrew G. Barto'
    assert judges.name_tokens(bullets) == {"richard", "sutton", "andrew", "barto"}
    numbered = "- 1. Fernando G. O. Figueiredo 2. Jian Yang 3. Mei Li"
    assert judges.name_tokens(numbered) == {"fernando", "figueiredo", "jian", "yang", "mei", "li"}
    llama = 'Sure, here are the author names of the reference "provided reference":\n\n1.Michael T.Taylor\n2.Andrew H.Bell'
    assert judges.name_tokens(llama) == {"michael", "taylor", "andrew", "bell"}
    # name order does not matter
    assert judges.name_tokens("Bishop, Christopher") == judges.name_tokens("Christopher M. Bishop")
    assert judges.name_tokens("") == set() and judges.name_tokens("-") == set()
    assert judges.name_tokens("J., K.") == set()               # initials only
    # refusal sentences (pilot, gpt-4) must not yield name tokens
    for refusal in ["I am unable to find the specific reference you mentioned. Please provide more information",
                    "Unfortunately, I cannot find the exact reference you are looking for. It may be"]:
        assert judges.name_tokens(refusal) == set(), refusal


def test_j_str_bounds():
    same = ["Ann Lee, Bo Chen"] * 3
    assert judges.j_str(same) == 0.0
    assert judges.j_str(["Ann Lee", "Bo Chen", "Cy Park"]) == 1.0
    assert judges.j_str(["", "", ""]) == 1.0                 # empty = maximally inconsistent


def test_j_nli_empty_answer_rule_and_pairs():
    calls = []

    def fake(prem, hyp):
        calls.append(len(prem))
        return np.zeros(len(prem))                            # never contradicts
    names = ["Ann Lee", "Bo Chen", "Cy Park"]
    assert judges.j_nli(names, fake) == (0.0, 0.0) and calls == [6]
    # one empty answer: 4 of the 6 ordered pairs involve it -> scored 1 under both variants
    rule, raw = judges.j_nli(["Ann Lee", "", "Cy Park"], fake)
    assert abs(rule - 4 / 6) < 1e-12 and abs(raw - 4 / 6) < 1e-12
    # refusals: the raw variant lets two refusals 'agree'; the names rule does not
    refusal = "I could not find a reference with that title."
    rule, raw = judges.j_nli([refusal, refusal, "Ann Lee"], fake)
    assert raw == 0.0 and rule == 1.0


def test_cluster_bootstrap_is_paired_and_unflipped():
    y = np.array([0, 1] * 50)
    topics = np.repeat(np.arange(20), 5)
    good, bad = y + 0.1 * np.arange(100) % 1, 1 - y
    res = cluster_bootstrap_auc({"good": good, "bad": bad}, y, topics, n_boot=200, seed=1)
    assert res["good"]["auc"] == 1.0 and res["bad"]["auc"] == 0.0     # no auto-flip
    d = paired_delta(res, "good", "bad")
    assert d["delta"] == 1.0 and d["lo"] == 1.0


def test_e1_cache_merge_with_placeholder_columns(tmp=Path(__import__("tempfile").mkdtemp())):
    """Regression (2026-11-04): E1 crashed when `part` already had j_nli columns before the merge."""
    import importlib.util

    import pandas as pd
    spec = importlib.util.spec_from_file_location("e1", ROOT / "scripts/12_e1_judge_swap.py")
    e1 = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(e1)
    e1.ROOT = tmp
    (tmp / "results/raw").mkdir(parents=True)
    pd.DataFrame({"row": [0, 1], "j_nli": [0.1, 0.9], "j_nli_raw": [0.1, 0.8]}).to_csv(
        tmp / "results/raw/e1_nli_m_main.csv", index=False)
    df = pd.DataFrame({"model": ["m", "m"], "row": [1, 0], "iq_1": ["a", "a"], "iq_2": ["a", "a"],
                       "iq_3": ["a", "a"], "j_nli": [None, None], "j_nli_raw": [None, None]})
    out = e1.nli_scores(df, "m", "main", scorer=None)   # everything cached: scorer never called
    assert out["j_nli"].tolist() == [0.9, 0.1] and out["j_nli_raw"].tolist() == [0.8, 0.1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
