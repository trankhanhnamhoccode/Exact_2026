from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xai_physics.core.result import SolveResult
from xai_physics.schema_solver import solve_schema


@dataclass
class SchemaCandidate:
    source: str
    schema: dict[str, Any]
    base_score: float = 0.0


@dataclass
class CandidateSelection:
    candidate: SchemaCandidate
    solve_result: SolveResult
    score: float
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


def _objects(schema: dict[str, Any]) -> list[dict[str, Any]]:
    objs = schema.get("objects", [])
    return objs if isinstance(objs, list) else []


def _queries(schema: dict[str, Any]) -> list[dict[str, Any]]:
    raw = schema.get("queries")
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return [obj for obj in _objects(schema) if obj.get("role") == "query"]


def _query_types(schema: dict[str, Any]) -> set[str]:
    return {str(q.get("type")) for q in _queries(schema) if q.get("type")}


def _has_null_given(schema: dict[str, Any]) -> bool:
    for obj in _objects(schema):
        if obj.get("role") in {"given", "constant"} and obj.get("value") is None:
            return True
    return False


def _intent_score(problem: str, schema: dict[str, Any]) -> float:
    low = problem.lower()
    qtypes = _query_types(schema)
    score = 0.0

    asks_voltage = any(p in low for p in ["calculate the voltage", "find the voltage", "determine the voltage", "what is the voltage"])
    asks_charge = (
        "charge" in low
        and any(p in low for p in ["calculate", "what", "find", "determine", "stored", "accumulated", "maximum"])
        and not asks_voltage
    ) or bool(__import__("re").search(r"\b(?:find|determine|calculate)\s+(?:the\s+)?(?:value\s+of\s+)?q\s*\d*\b", low))
    asks_capacitance = (
        "capacitance" in low and any(p in low for p in ["calculate", "what", "find", "determine"]) and not asks_voltage
    ) or bool(__import__("re").search(r"\b(?:find|determine|calculate)\s+c\s*(?:'|prime)?\b", low))
    if asks_capacitance:
        asks_charge = False

    if asks_voltage:
        score += 60 if "voltage" in qtypes else -80
    if any(p in low for p in ["dielectric constant", "relative permittivity", "permittivity"]):
        score += 40 if "relative_permittivity" in qtypes else -90
    if asks_charge:
        score += 25 if "charge" in qtypes else -35
    if asks_capacitance:
        if not any(p in low for p in ["dielectric constant", "relative permittivity"]):
            score += 25 if "capacitance" in qtypes else -35
    if "energy" in low:
        score += 25 if "energy" in qtypes else -35
    if any(p in low for p in ["force acting", "net electric force", "net force", "resultant force"]):
        score += 40 if ("net_force" in qtypes or "resultant_vector" in qtypes or "force" in qtypes) else -80
    zero_field_location = (
        "zero" in low
        and any(p in low for p in ["electric field", "field strength", "resultant electric field", "net electric field"])
        and any(p in low for p in ["where", "point", "coordinate", "distance"])
    )
    if zero_field_location:
        score += 65 if qtypes & {"distance", "position"} else -70
    elif "electric field" in low and not asks_charge and not any(p in low for p in ["electric field energy", "energy stored"]):
        score += 30 if "electric_field" in qtypes else -40

    return score


def _source_score(source: str) -> float:
    return {
        "formula_driven": 22.0,
        "deterministic_text": 18.0,
        "llm_repaired": 12.0,
        "llm_raw": 5.0,
    }.get(source, 0.0)


def score_candidate(problem: str, candidate: SchemaCandidate, result: SolveResult) -> tuple[float, dict[str, Any]]:
    score = candidate.base_score + _source_score(candidate.source)
    diag: dict[str, Any] = {"source": candidate.source, "status": result.status, "answer": result.answer}

    if result.status in {"ok", "solved"} and result.answer is not None:
        score += 100
    else:
        score -= 250
        diag["solve_error"] = result.error

    if _has_null_given(candidate.schema):
        score -= 80
        diag["null_given_penalty"] = True

    intent = _intent_score(problem, candidate.schema)
    score += intent
    diag["intent_score"] = intent
    diag["query_types"] = sorted(_query_types(candidate.schema))
    diag["score"] = score
    return score, diag


def select_best_candidate(problem: str, candidates: list[SchemaCandidate]) -> CandidateSelection | None:
    best: CandidateSelection | None = None
    diagnostics: list[dict[str, Any]] = []

    for candidate in candidates:
        try:
            result = solve_schema(candidate.schema)
        except Exception as exc:  # defensive; solve_schema should normally return SolveResult
            result = SolveResult(status="solve_failed", domain=str(candidate.schema.get("domain", "unknown")), error=f"{type(exc).__name__}: {exc}")

        score, diag = score_candidate(problem, candidate, result)
        diagnostics.append(diag)
        if best is None or score > best.score:
            best = CandidateSelection(candidate=candidate, solve_result=result, score=score)

    if best is not None:
        best.diagnostics = diagnostics
    return best
