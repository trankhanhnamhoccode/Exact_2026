from __future__ import annotations

from collections import Counter
import math
import re


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[????]")


SYNONYMS = {
    "u": ["voltage"],
    "v": ["voltage"],
    "potential": ["voltage"],
    "difference": ["voltage"],
    "c": ["capacitance"],
    "q": ["charge"],
    "i": ["current"],
    "r": ["resistance", "distance"],
    "l": ["inductance", "length"],
    "b": ["magnetic", "field"],
    "e": ["electric", "field"],
    "omega": ["angular", "frequency"],
    "resonant": ["resonance"],
    "oscillation": ["lc", "resonance"],
    "oscillating": ["lc", "resonance"],
    "stored": ["energy"],
}


def tokenize(text: str) -> list[str]:
    raw_tokens = [tok.lower() for tok in TOKEN_RE.findall(text)]
    tokens: list[str] = []

    for tok in raw_tokens:
        if tok in {"?", "?"}:
            tok = "u"
        if tok in {"?", "?"}:
            tok = "ohm"

        tokens.append(tok)
        tokens.extend(SYNONYMS.get(tok, []))

    return tokens


def embed_text(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine_score(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0

    common = set(a) & set(b)
    dot = sum(a[tok] * b[tok] for tok in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def similarity(query: str, document: str) -> float:
    return cosine_score(embed_text(query), embed_text(document))
