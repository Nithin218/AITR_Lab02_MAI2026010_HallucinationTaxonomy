"""Topic lists written by scripts/01_make_pilot_split.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def topics(split: str) -> list[str]:
    """split: 'pilot' | 'main' | 'e2'."""
    path = ROOT / f"configs/{split}_topics.txt"
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
