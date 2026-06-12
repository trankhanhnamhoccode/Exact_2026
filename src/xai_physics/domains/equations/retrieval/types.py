from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TagHit:
    tag: str
    score: float
    evidence: str


@dataclass(frozen=True)
class FormulaDoc:
    id: str
    name: str
    equation: str
    description: str
    quantity_types: list[str]
    query_types: list[str]
    tags: list[str]
    keywords: list[str]
    schema_template: dict[str, Any]

    def searchable_text(self) -> str:
        return " ".join(
            [
                self.id,
                self.name,
                self.equation,
                self.description,
                " ".join(self.quantity_types),
                " ".join(self.query_types),
                " ".join(self.tags),
                " ".join(self.keywords),
            ]
        )


@dataclass(frozen=True)
class ExampleDoc:
    id: str
    problem: str
    formula_id: str
    tags: list[str]
    schema: dict[str, Any]

    def searchable_text(self) -> str:
        return " ".join([self.id, self.problem, self.formula_id, " ".join(self.tags)])


@dataclass(frozen=True)
class RetrievedFormula:
    formula: FormulaDoc
    vector_score: float
    rule_score: float
    tag_score: float
    rerank_score: float
    matched_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedExample:
    example: ExampleDoc
    vector_score: float
    formula_score: float
    tag_score: float
    rerank_score: float
    matched_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EquationRetrievalResult:
    selected_formulas: list[RetrievedFormula]
    selected_examples: list[RetrievedExample]
    final_tags: list[str]
    debug: dict[str, Any]
