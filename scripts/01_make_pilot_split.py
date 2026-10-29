"""Split the 200 ACM CCS topics into 10 pilot and 190 main topics (protocol §4, seed 2026),
and draw the E2 subset of main topics (D-013).

Usage (repo root):  python scripts/01_make_pilot_split.py
Writes configs/pilot_topics.txt, configs/main_topics.txt and configs/e2_topics.txt.
"""
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aitr.download import fetch_agrawal  # noqa: E402

cfg = yaml.safe_load((ROOT / "configs/protocol_v1.yaml").read_text())
raw = ROOT / "data/raw/agrawal"
sums = fetch_agrawal(raw, cfg["data"]["agrawal_commit"], ["acm_ccs_200.titles"])

topics = [t.strip() for t in (raw / "acm_ccs_200.titles").read_text(encoding="utf-8").splitlines() if t.strip()]
assert len(topics) == 200, f"expected 200 topics, got {len(topics)}"
assert len(set(topics)) == 200, "duplicate topics in acm_ccs_200.titles"

pilot = random.Random(cfg["seeds"]["pilot_split"]).sample(topics, cfg["data"]["n_pilot_topics"])
main = [t for t in topics if t not in set(pilot)]

header = f"# source acm_ccs_200.titles sha256={sums['acm_ccs_200.titles']} seed={cfg['seeds']['pilot_split']}\n"
(ROOT / "configs/pilot_topics.txt").write_text(header + "\n".join(pilot) + "\n", encoding="utf-8")
(ROOT / "configs/main_topics.txt").write_text(header + "\n".join(main) + "\n", encoding="utf-8")
# Separate generator so adding this step leaves the pilot draw above unchanged.
e2 = random.Random(cfg["seeds"]["e2_subset"]).sample(main, cfg["data"]["n_e2_topics"])
(ROOT / "configs/e2_topics.txt").write_text(
    header.replace(f"seed={cfg['seeds']['pilot_split']}", f"e2_seed={cfg['seeds']['e2_subset']} (subset of main)")
    + "\n".join(e2) + "\n", encoding="utf-8")
print(f"pilot={len(pilot)} main={len(main)} e2={len(e2)} sha256={sums['acm_ccs_200.titles'][:12]}")
print("pilot topics:", *pilot, sep="\n  ")
