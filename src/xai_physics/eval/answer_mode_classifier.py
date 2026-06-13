from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

AnswerMode = Literal[
    "numeric",
    "numeric_multi",
    "symbolic_expr",
    "symbolic_relation",
    "direction",
    "formula_text",
    "qualitative",
    "empty",
]

_NUMBER_RE = re.compile(r"[-+]?\d+(?:[\.,]\d+)?(?:\s*(?:×|x|\*)\s*10\s*\^?\s*[-+]?\d+|[eE][-+]?\d+)?")
_SYMBOL_RE = re.compile(r"(?:\\sqrt|sqrt|√|π|\\pi|\bpi\b|F[₀0]|E[₁₂12]|I[₀₁₂0-9]|W[CL₀0]?|q[₀₁₂₃0-9]?|\bk\b|\ba\b\s*(?:\^|²))")
_FORMULA_RE = re.compile(r"[A-Za-z][A-Za-z₀-₉0-9_]*\s*=|[½¼¾]|\bsin\b|\bcos\b|\btan\b", re.I)
_DIRECTION_RE = re.compile(r"hướng|huong|toward|direction", re.I)
_QUAL_WORD_RE = re.compile(
    r"\b(?:increase|decrease|brighter|maximum|minimum|zero|unchanged|equal|less than|greater than|stored|field|parabola|linear|joule|doubled|triple|half|conservation|resistance)\b",
    re.I,
)


@dataclass(frozen=True)
class AnswerModeInfo:
    answer_mode: AnswerMode
    reason: str


def classify_expected_answer(answer: object, unit: object | None = None) -> AnswerModeInfo:
    text = "" if answer is None else str(answer).strip()
    unit_text = "" if unit is None else str(unit).strip()
    if not text:
        return AnswerModeInfo("empty", "blank_answer")

    low = text.lower()
    number_count = len(_NUMBER_RE.findall(text))
    has_number = number_count > 0
    unit_parts = [part.strip() for part in re.split(r";|,", unit_text) if part.strip()]

    if _DIRECTION_RE.search(text) or "phía" in low:
        return AnswerModeInfo("direction", "directional_text")

    # A semicolon usually means multiple labeled numeric outputs in this dataset:
    # e.g. I_D1 = 1.0; I_D2 = 1.0; I_total = 2.0 with units A; A; A.
    if ";" in text or len(unit_parts) > 1:
        if has_number:
            return AnswerModeInfo("numeric_multi", "semicolon_or_multi_unit_numeric")
        return AnswerModeInfo("qualitative", "semicolon_or_multi_unit_text")

    if "=" in text:
        # Relation answers with variables on both sides should not be judged as a
        # single numeric value.  Labeled numeric outputs such as P = 48.0 are
        # still numeric when a physical unit is supplied.
        has_symbolic_token = bool(_SYMBOL_RE.search(text))
        if has_symbolic_token and unit_text in {"", "-", "—"}:
            return AnswerModeInfo("symbolic_relation", "symbolic_equality_or_ratio")
        if has_number and unit_text not in {"", "-", "—"}:
            return AnswerModeInfo("numeric", "labeled_numeric_with_unit")
        if _FORMULA_RE.search(text):
            return AnswerModeInfo("formula_text", "formula_like_text")

    if _SYMBOL_RE.search(text):
        # sqrt with pure numbers and a physical unit can still be numeric-like,
        # e.g. 9sqrt(3)e-27 N.  Keep it symbolic_expr for reporting, while the
        # evaluator can still compare numerically when implemented.
        return AnswerModeInfo("symbolic_expr", "symbolic_token_or_radical")

    if has_number:
        return AnswerModeInfo("numeric", "numeric_literal")

    if _FORMULA_RE.search(text):
        return AnswerModeInfo("formula_text", "formula_like_text")

    if _QUAL_WORD_RE.search(text) or unit_text in {"-", "—", ""}:
        return AnswerModeInfo("qualitative", "qualitative_text")

    return AnswerModeInfo("qualitative", "fallback_text")
