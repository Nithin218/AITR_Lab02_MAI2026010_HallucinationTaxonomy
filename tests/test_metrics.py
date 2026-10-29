"""Tests for src/aitr/metrics.py. Run: python tests/test_metrics.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr.metrics import agrawal_bootstrap_auc, point_auc  # noqa: E402

Y = np.array([0] * 50 + [1] * 50)


def test_perfect_and_inverted():
    s = Y.astype(float)
    assert point_auc(s, Y) == 1.0
    assert agrawal_bootstrap_auc(s, Y)["auc_mean"] == 1.0
    inv = agrawal_bootstrap_auc(1 - s, Y)                    # inverted predictor
    assert inv["flipped"] and inv["auc_mean"] == 1.0         # Agrawal-style auto-flip hides it
    raw = agrawal_bootstrap_auc(1 - s, Y, auto_flip=False)   # our analyses: report it as it is
    assert not raw["flipped"] and raw["auc_mean"] == 0.0


def test_constant_score_is_half():
    # mirrors Llama-2-7B DQ1, where every answer was "yes" (published 0.500 ± nan)
    assert agrawal_bootstrap_auc(np.ones(100), Y)["auc_mean"] == 0.5


def test_near_ties_are_merged():
    # 1 - 0.8000000000000002 and 0.2 must be one threshold, as in roc-utils (round 8)
    s = np.array([1 - 0.8000000000000002, 0.2])
    assert point_auc(s, [0, 1]) == 0.5


def test_deterministic_and_close_to_point_auc():
    rng = np.random.default_rng(0)
    s = Y + rng.normal(0, 1.0, size=len(Y))
    a, b = agrawal_bootstrap_auc(s, Y), agrawal_bootstrap_auc(s, Y)
    assert a == b
    assert abs(a["auc_mean"] - point_auc(s, Y)) < 0.02


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
