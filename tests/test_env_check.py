"""Tests for scripts/00_env_check.split_conflicts. Run: python -m pytest tests/ (or python tests/test_env_check.py)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.argv = ["test"]
spec = importlib.util.spec_from_file_location("env_check", ROOT / "scripts/00_env_check.py")
env_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env_check)

# Verbatim `pip check` output from the Colab run on 2026-10-29 (env_verification_log.md, Run 1).
COLAB_OUTPUT = """ipython 7.34.0 requires jedi, which is not installed.
gradio 6.30.0 has requirement huggingface-hub<3.0,>=1.16.0, but you have huggingface-hub 0.36.2.
gradio-client 2.7.2 has requirement huggingface-hub<3.0,>=1.16.0, but you have huggingface-hub 0.36.2.
diffusers 0.42.0 has requirement huggingface-hub<3.0,>=1.32.0, but you have huggingface-hub 0.36.2."""


def test_colab_conflicts_are_unrelated():
    relevant, ignored = env_check.split_conflicts(COLAB_OUTPUT, env_check.REQUIRED + env_check.GPU_ONLY)
    assert relevant == []
    assert len(ignored) == 4


def test_conflict_in_our_package_is_reported():
    # edge case: a requirement declared by one of OUR packages, with underscore/dash spelling differences
    text = "bert-score 0.3.13 has requirement torch>=99, but you have torch 2.11.0.\nselfcheckgpt 0.1.7 requires groq, which is not installed."
    relevant, ignored = env_check.split_conflicts(text, env_check.REQUIRED + env_check.GPU_ONLY)
    assert len(relevant) == 2 and ignored == []


def test_clean_output():
    assert env_check.split_conflicts("No broken requirements found.", env_check.REQUIRED) == ([], [])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
