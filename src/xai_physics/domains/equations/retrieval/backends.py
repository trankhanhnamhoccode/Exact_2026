from __future__ import annotations

from typing import Protocol

from .types import ExampleDoc, FormulaDoc, TagHit
from .vector_store import similarity


class EmbeddingBackend(Protocol):
    def score(self, query: str, document: str) -> float:
        ...


class RerankerBackend(Protocol):
    def score_formula(
        self,
        query: str,
        formula: FormulaDoc,
        *,
        vector_score: float,
        rule_score: float,
        tag_score: float,
        matched_tags: list[str],
        tag_hits: list[TagHit],
    ) -> float:
        ...

    def score_example(
        self,
        query: str,
        example: ExampleDoc,
        *,
        vector_score: float,
        formula_score: float,
        tag_score: float,
        matched_tags: list[str],
        tag_hits: list[TagHit],
    ) -> float:
        ...


class LexicalEmbeddingBackend:
    """
    Default lightweight embedding backend.

    Future replacement:
        SentenceTransformerEmbeddingBackend
        OpenAIEmbeddingBackend
        LocalFaissEmbeddingBackend
    """

    def score(self, query: str, document: str) -> float:
        return similarity(query, document)


class HeuristicRerankerBackend:
    """
    Default transparent reranker.

    Future replacement:
        CrossEncoderRerankerBackend
        LLMRerankerBackend
        LearnedFormulaClassifierBackend
    """

    def score_formula(
        self,
        query: str,
        formula: FormulaDoc,
        *,
        vector_score: float,
        rule_score: float,
        tag_score: float,
        matched_tags: list[str],
        tag_hits: list[TagHit],
    ) -> float:
        return (
            0.55 * vector_score
            + 1.00 * rule_score
            + 0.25 * tag_score
        )

    def score_example(
        self,
        query: str,
        example: ExampleDoc,
        *,
        vector_score: float,
        formula_score: float,
        tag_score: float,
        matched_tags: list[str],
        tag_hits: list[TagHit],
    ) -> float:
        return (
            0.60 * vector_score
            + 1.00 * formula_score
            + 0.20 * tag_score
        )
