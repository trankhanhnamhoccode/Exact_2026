from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AnswerType = Literal["numeric", "symbolic", "relation", "direction", "quality_flag", "text"]


@dataclass(frozen=True)
class AnswerEnvelope:
    """Structured answer metadata kept next to the legacy display string.

    The public ``SolveResult.answer`` field stays unchanged for backward
    compatibility.  This envelope lets evaluators/reporters decide whether the
    answer should be compared as a number, symbolic expression, symbolic
    relation, direction, text, or quality flag.
    """

    answer_type: AnswerType
    display: str
    unit: str | None = None
    quantity_type: str | None = None
    numeric_value_si: float | None = None
    display_value: float | None = None
    symbolic_canonical: tuple[Any, ...] | None = None
    variables: dict[str, str] = field(default_factory=dict)
    relation: dict[str, Any] = field(default_factory=dict)
    direction: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    formula: str | None = None
    confidence: float | None = None
    reason: str | None = None

    @staticmethod
    def numeric(
        *,
        display: str,
        value_si: float,
        display_value: float,
        unit: str | None,
        quantity_type: str,
        formula: str | None = None,
        source: str | None = "formula_solver",
        confidence: float | None = None,
    ) -> "AnswerEnvelope":
        return AnswerEnvelope(
            answer_type="numeric",
            display=display,
            unit=unit,
            quantity_type=quantity_type,
            numeric_value_si=float(value_si),
            display_value=float(display_value),
            formula=formula,
            source=source,
            confidence=confidence,
        )

    @staticmethod
    def symbolic(
        *,
        display: str,
        canonical: tuple[Any, ...] | None,
        unit: str | None = None,
        variables: dict[str, str] | None = None,
        formula: str | None = None,
        source: str | None = "symbolic_solver",
        confidence: float | None = None,
    ) -> "AnswerEnvelope":
        return AnswerEnvelope(
            answer_type="symbolic",
            display=display,
            unit=unit,
            symbolic_canonical=canonical,
            variables=variables or {},
            formula=formula,
            source=source,
            confidence=confidence,
        )

    @staticmethod
    def relation_answer(
        *,
        display: str,
        canonical: tuple[Any, ...] | None,
        relation: dict[str, Any],
        unit: str | None = None,
        formula: str | None = None,
        source: str | None = "symbolic_solver",
        confidence: float | None = None,
    ) -> "AnswerEnvelope":
        return AnswerEnvelope(
            answer_type="relation",
            display=display,
            unit=unit,
            symbolic_canonical=canonical,
            relation=relation,
            formula=formula,
            source=source,
            confidence=confidence,
        )

    @staticmethod
    def direction_answer(
        *,
        display: str,
        canonical: tuple[Any, ...] | None,
        target: str,
        formula: str | None = None,
        source: str | None = "symbolic_solver",
        confidence: float | None = None,
    ) -> "AnswerEnvelope":
        return AnswerEnvelope(
            answer_type="direction",
            display=display,
            symbolic_canonical=canonical,
            direction={"target": target},
            formula=formula,
            source=source,
            confidence=confidence,
        )

    @staticmethod
    def quality_flag(
        *,
        display: str,
        reason: str,
        formula: str | None = None,
        source: str | None = "quality_gate",
    ) -> "AnswerEnvelope":
        return AnswerEnvelope(
            answer_type="quality_flag",
            display=display,
            reason=reason,
            formula=formula,
            source=source,
        )

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): AnswerEnvelope._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [AnswerEnvelope._jsonable(v) for v in value]
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return str(value)
        return value

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer_type": self.answer_type,
            "display": self.display,
            "unit": self.unit,
            "quantity_type": self.quantity_type,
            "numeric_value_si": self.numeric_value_si,
            "display_value": self.display_value,
            "symbolic_canonical": self._jsonable(self.symbolic_canonical),
            "variables": self.variables,
            "relation": self.relation,
            "direction": self.direction,
            "source": self.source,
            "formula": self.formula,
            "confidence": self.confidence,
            "reason": self.reason,
        }
