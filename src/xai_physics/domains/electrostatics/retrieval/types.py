from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TagHit:
    tag: str
    source: str
    score: float
    evidence: str


@dataclass(frozen=True)
class SchemaExample:
    id: str
    domain: str
    tags: list[str]
    problem: str
    schema: dict[str, Any]

    def searchable_text(self) -> str:
        return " ".join([self.id, self.problem, " ".join(self.tags)])


@dataclass(frozen=True)
class RetrievedExample:
    example: SchemaExample
    vector_score: float
    rule_score: float
    rerank_score: float
    matched_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ElectrostaticsRetrievalResult:
    domain: str
    hard_tags: list[TagHit]
    final_tags: list[str]
    selected_examples: list[RetrievedExample]
    debug: dict[str, Any] = field(default_factory=dict)
