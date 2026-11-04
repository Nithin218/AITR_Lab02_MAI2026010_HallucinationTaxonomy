"""Tests for src/aitr/prompts.py. Run: python tests/test_prompts.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr import prompts  # noqa: E402


def test_prompts_match_upstream_verbatim():
    # docs/upstream/agrawal_prompts.py is the verbatim upstream file
    ns = {}
    exec((ROOT / "docs/upstream/agrawal_prompts.py").read_text(), ns)
    for name in ["prompt_Q", "prompt_IQ", "prompt_DQ1_sample", "prompt_DQ2_sample"]:
        assert getattr(prompts, name)("T") == ns[name]("T"), name
    assert prompts.prompt_DQ3_sample("A\nB", 2) == ns["prompt_DQ3_sample"]("A\nB", 2)


def test_extract_titles():
    quoted = '1. "Deep Learning"\n2. "Pattern Recognition and Machine Learning"\n3. “Reinforcement Learning”'
    assert prompts.extract_titles(quoted) == (
        ["Deep Learning", "Pattern Recognition and Machine Learning", "Reinforcement Learning"], "agrawal")
    md = "Here are five:\n\n1. **Data Cleaning: Problems and Approaches**\n2. Data Quality."
    assert prompts.extract_titles(md) == (["Data Cleaning: Problems and Approaches", "Data Quality"], "fallback")
    assert prompts.extract_titles("no list here") == ([], "fallback")


def test_agreement_and_yes():
    assert abs(prompts.get_agreement_frac("ANS: 66.67 JUSTIFICATION: ...") - 0.6667) < 1e-9
    assert prompts.get_agreement_frac("Answer: 150") == 1.0
    assert prompts.get_agreement_frac("I cannot tell") == 0.0
    assert prompts.dq_yes("Yes, it exists.") and not prompts.dq_yes("No.")
    assert prompts.parse_verbal("85") == 85.0 and prompts.parse_verbal("none") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
