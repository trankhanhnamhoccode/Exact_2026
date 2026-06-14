from __future__ import annotations

from typing import Any

from xai_physics.core.result import SolveResult
from xai_physics.domains.capacitor_state.engine import solve_schema as solve_capacitor_schema
from xai_physics.domains.electrostatics.continuous import solve_schema as solve_continuous_electrostatics_schema
from xai_physics.domains.electrostatics.engine import solve_schema as solve_electrostatics_schema
from xai_physics.domains.equations.solver import solve_schema as solve_equations_schema
from xai_physics.domains.symbolic_geometry.engine import solve_schema as solve_symbolic_geometry_schema


SUPPORTED_DOMAINS = {
    "capacitor_state",
    "electrostatics",
    "equations",
    "symbolic_geometry",
}


def solve_schema(schema: dict[str, Any]) -> SolveResult:
    """
    Unified schema entrypoint.

    This is the main function that future LLM/extractor output should call:
        raw text -> canonical schema -> solve_schema(schema)
    """
    if not isinstance(schema, dict):
        return SolveResult(
            status="solve_failed",
            domain="unknown",
            error="Schema must be a dict/object.",
        )

    domain = schema.get("domain")

    if domain == "capacitor_state":
        return solve_capacitor_schema(schema)

    if domain == "electrostatics":
        representation = str(schema.get("representation") or schema.get("answer_representation") or "").lower()
        solver_backend = str(schema.get("solver_backend") or schema.get("subtype") or "").lower()
        if representation == "symbolic" and solver_backend in {"symbolic_geometry", "symbolic_relation", "geometry"}:
            return solve_symbolic_geometry_schema(schema)
        if solver_backend in {"symbolic_geometry", "symbolic_relation"}:
            return solve_symbolic_geometry_schema(schema)
        if solver_backend in {"continuous_distribution", "continuous_electrostatics", "sheet_field"}:
            return solve_continuous_electrostatics_schema(schema)
        return solve_electrostatics_schema(schema)

    if domain == "equations":
        return solve_equations_schema(schema)

    if domain == "symbolic_geometry":
        return solve_symbolic_geometry_schema(schema)

    return SolveResult(
        status="solve_failed",
        domain=str(domain),
        error=f"Unsupported domain: {domain!r}. Supported domains: {sorted(SUPPORTED_DOMAINS)}",
    )
