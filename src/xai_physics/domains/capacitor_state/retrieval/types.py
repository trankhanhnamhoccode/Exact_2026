from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TagHit:
    tag: str
    source: str
    score: float
    evidence: str


@dataclass
class SchemaExample:
    id: str
    domain: str
    tags: list[str]
    problem: str
    schema: dict[str, Any]


@dataclass
class RetrievedExample:
    example: SchemaExample
    vector_score: float
    rule_score: float
    rerank_score: float
    matched_tags: list[str] = field(default_factory=list)


@dataclass
class CapacitorRetrievalResult:
    domain: str
    hard_tags: list[TagHit]
    soft_tags: list[TagHit]
    final_tags: list[str]
    selected_examples: list[RetrievedExample]
    debug: dict[str, Any] = field(default_factory=dict)
