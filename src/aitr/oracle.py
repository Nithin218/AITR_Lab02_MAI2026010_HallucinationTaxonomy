"""Scholarly-database oracle: OpenAlex first, then Crossref (protocol §4). Responses are cached on disk."""
import json
import time
from pathlib import Path

import requests
from rapidfuzz import fuzz


def norm_title(s: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in s or "").split())


def similarity(a: str, b: str) -> float:
    """Order-sensitive: token_sort_ratio matched reordered fake titles on the pilot (D-017)."""
    return float(fuzz.ratio(norm_title(a), norm_title(b)))


class Oracle:
    def __init__(self, cache_path: Path, mailto: str = "", per_page: int = 5):
        self.cache_path = Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.cache[(rec["source"], rec["query"])] = rec["titles"]
        self.mailto, self.per_page = mailto, per_page
        ua = "AITR-Lab02 research script" + (f" (mailto:{mailto})" if mailto else "")
        self.session = requests.Session()
        self.session.headers["User-Agent"] = ua

    def _get(self, url, params, pause):
        for attempt in range(8):
            r = self.session.get(url, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                wait = r.headers.get("Retry-After", "")
                time.sleep(float(wait) if wait.isdigit() else min(2 ** attempt * 2, 120))
                continue
            r.raise_for_status()
            time.sleep(pause)
            return r.json()
        r.raise_for_status()

    def _cached(self, source, query, fetch):
        key = (source, norm_title(query))
        if key not in self.cache:
            titles = fetch(query)
            self.cache[key] = titles
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"source": source, "query": key[1], "titles": titles}, ensure_ascii=False) + "\n")
        return self.cache[key]

    def openalex(self, title: str) -> list[str]:
        params = {"search": title, "per_page": self.per_page} | ({"mailto": self.mailto} if self.mailto else {})
        return self._cached("openalex", title, lambda q: [
            w.get("display_name") or "" for w in self._get("https://api.openalex.org/works", params, 0.15)["results"]])

    def crossref(self, title: str) -> list[str]:
        params = {"query.bibliographic": title, "rows": self.per_page} | ({"mailto": self.mailto} if self.mailto else {})
        pause = 0.15 if self.mailto else 0.3  # public pool is slower (Crossref docs)
        return self._cached("crossref", title, lambda q: [
            (w.get("title") or [""])[0] for w in self._get("https://api.crossref.org/works", params, pause)["message"]["items"]])

    def label(self, title: str, tau: float, partial_band: float) -> dict:
        """found / partial / not_found. Crossref is queried only if OpenAlex has no match >= tau."""
        oa = max([(similarity(title, c), c) for c in self.openalex(title)] or [(0.0, "")])
        best, source = oa, "openalex"
        if oa[0] < tau:
            cr = max([(similarity(title, c), c) for c in self.crossref(title)] or [(0.0, "")])
            if cr[0] > best[0]:
                best, source = cr, "crossref"
        status = "found" if best[0] >= tau else "partial" if best[0] >= tau - partial_band else "not_found"
        return {"status": status, "best_sim": round(best[0], 1), "best_title": best[1], "source": source}
