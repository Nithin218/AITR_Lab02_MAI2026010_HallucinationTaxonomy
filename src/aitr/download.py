"""Download the released Agrawal et al. data at a pinned commit and record checksums."""
import hashlib
from pathlib import Path

import requests

AGRAWAL_RAW = "https://raw.githubusercontent.com/microsoft/hallucinated-references/{commit}/data/{name}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_agrawal(dest_dir: Path, commit: str, names: list[str], force: bool = False) -> dict[str, str]:
    """Download each file to dest_dir (skipping existing ones unless force) and return {name: sha256}."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name in names:
        path = dest_dir / name
        if force or not path.exists():
            resp = requests.get(AGRAWAL_RAW.format(commit=commit, name=name), timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        checksums[name] = sha256(path)
    return checksums
