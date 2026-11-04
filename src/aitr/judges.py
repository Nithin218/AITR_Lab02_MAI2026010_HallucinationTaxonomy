"""Author-consistency judges for the E1 judge swap (protocol §3).

All judges score one reference from its 3 sampled author answers; higher = more likely hallucinated.
  J-LM   released LM-judged agreement (Agrawal et al.), score = 1 - iq_lm      (computed by the caller)
  J-NLI  mean over the 6 ordered pairs (i != j) of P(contradiction | premise=answer_i, hypothesis=answer_j)
  J-STR  1 - mean pairwise Jaccard of normalised name-token sets (no model)
Empty answers count as maximally inconsistent: every pair involving one scores 1 (NLI) / Jaccard 0 (STR).
"""
import re
import unicodedata
from itertools import combinations, permutations

import numpy as np

# Chunks containing these words are chat boilerplate, not names ("Sure, here are the authors...").
# Rules fixed on pilot topics only (research_decisions.md D-017).
_BOILERPLATE = re.compile(
    r"\b(sure|here|author|authors|reference|references|provided|name|names|listed|following|"
    r"apologi[sz]e|unable|cannot|sorry|note|information|title|paper|book|et al|"
    # function words that occur in refusal sentences but not in author names
    r"i|i'm|you|it|is|are|was|be|am|could|would|can|may|might|please|find|found|the|this|that|with|"
    r"for|not|any|there|however|specific|exact|known|unfortunately|course|happy|help|certainly)\b", re.I)


def _strip_preamble(t: str) -> str:
    """Drop text before a colon while that text is boilerplate: 'Sure, here are the authors of ...: Ann Lee'."""
    while ":" in t:
        head, tail = t.split(":", 1)
        if not _BOILERPLATE.search(head):
            break
        t = tail
    return t


def name_chunks(text: str) -> list[list[str]]:
    """Split an author answer into candidate names, each a list of word tokens.

    Separators: commas, semicolons, newlines, 'and', '&', bullets (* - •) and list numbers ('1.', '2)'),
    which in the released Llama answers often appear mid-line.
    """
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    t = _strip_preamble(t)
    t = re.sub(r"(^|\s)\d+[.)]\s*", "\n", t)
    names = []
    for part in re.split(r",|;|\n|\band\b|&|\*|•|(?:^|\s)-(?:\s|$)", t):
        part = part.strip(" .\"'*-")
        if not part or _BOILERPLATE.search(part):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", part)
        if 1 <= len(words) <= 5:
            names.append(words)
    return names


def name_tokens(text: str) -> set[str]:
    """All name words of length >= 2, lowercased.

    Using every token (not just a surname) makes 'Bishop, Christopher' and 'Christopher M. Bishop'
    agree, since comma-separated 'Last, First' lists cannot be split into people reliably.
    """
    return {w.lower() for words in name_chunks(text) for w in words if len(w) >= 2}


def j_str(samples: list[str]) -> float:
    sets = [name_tokens(s) for s in samples]
    jac = [len(a & b) / len(a | b) if (a | b) else 0.0 for a, b in combinations(sets, 2)]
    return 1.0 - float(np.mean(jac))


def nli_pairs(samples: list[str]) -> list[tuple[int, int]]:
    """Ordered (premise, hypothesis) index pairs that need the NLI model (pairs with an empty answer do not)."""
    return [(i, j) for i, j in permutations(range(len(samples)), 2) if samples[i].strip() and samples[j].strip()]


def j_nli(samples: list[str], contradiction) -> tuple[float, float]:
    """Returns (names_rule, raw); contradiction(premises, hypotheses) -> P(contradiction), one call per reference.

    names_rule (primary, D-017): an answer with no parseable name (e.g. "I could not find that reference")
        counts as empty, so its pairs score 1. Agrawal et al.'s LM judge was given the same rule
        ("if either of them is not a list of authors, output 0").
    raw (sensitivity): only literally empty answers count as empty; two refusals can 'agree'.
    """
    pairs = list(permutations(range(len(samples)), 2))
    need = nli_pairs(samples)
    probs = dict(zip(need, contradiction([samples[i] for i, _ in need], [samples[j] for _, j in need]))) if need else {}
    raw = float(np.mean([probs.get(p, 1.0) for p in pairs]))
    has_names = [bool(name_tokens(s)) for s in samples]
    rule = float(np.mean([probs.get((i, j), 1.0) if has_names[i] and has_names[j] else 1.0 for i, j in pairs]))
    return rule, raw


class DebertaContradiction:
    """P(contradiction) from potsawee/deberta-v3-large-mnli, encoded like SelfCheckNLI: (hypothesis, premise)."""

    def __init__(self, name: str, device: str, max_length: int = 512, batch_size: int = 32):
        import torch
        from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
        self.torch, self.device, self.max_length, self.batch_size = torch, device, max_length, batch_size
        self.tok = DebertaV2Tokenizer.from_pretrained(name)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(name).eval().to(device)

    def __call__(self, premises, hypotheses) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for k in range(0, len(premises), self.batch_size):
                enc = self.tok(list(hypotheses[k:k + self.batch_size]), list(premises[k:k + self.batch_size]),
                               return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
                logits = self.model(**enc.to(self.device)).logits
                out.append(self.torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.array([])
