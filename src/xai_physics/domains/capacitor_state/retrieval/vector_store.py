from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from xai_physics.domains.capacitor_state.retrieval.types import SchemaExample


class EmbeddingModel(Protocol):
    def embed(self, text: str) -> dict[str, float]:
        ...


class TokenEmbeddingModel:
    """
    Lightweight local embedding baseline.

    This is intentionally simple and dependency-free.
    Later we can replace it with sentence-transformers or OpenAI embeddings
    without changing the retrieval pipeline.
    """

    def embed(self, text: str) -> dict[str, float]:
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower())
        counts = Counter(tok for tok in tokens if len(tok) >= 2)
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        return {tok: val / norm for tok, val in counts.items()}


def cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a

    return sum(val * b.get(key, 0.0) for key, val in a.items())


class VectorExampleIndex:
    def __init__(
        self,
        examples: list[SchemaExample],
        embedding_model: EmbeddingModel | None = None,
    ):
        self.examples = examples
        self.embedding_model = embedding_model or TokenEmbeddingModel()
        self.example_vectors = {
            ex.id: self.embedding_model.embed(
                ex.problem + " " + " ".join(ex.tags)
            )
            for ex in examples
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[SchemaExample, float]]:
        qvec = self.embedding_model.embed(query)

        scored = [
            (ex, cosine_sparse(qvec, self.example_vectors[ex.id]))
            for ex in self.examples
        ]

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
