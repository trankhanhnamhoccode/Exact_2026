from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xai_physics.core.result import SolveResult
from xai_physics.llm.client import SchemaLLMClient
from xai_physics.llm.json_extractor import extract_json_object
from xai_physics.llm.prompt_builder import PromptBuildResult, build_schema_prompt
from xai_physics.schema_solver import solve_schema
from xai_physics.domains.electrostatics.text_extractor import extract_electrostatics_schema_from_text
from xai_physics.domains.equations.text_extractor import extract_equations_schema_from_text
from xai_physics.hybrid.candidate_ranker import SchemaCandidate, select_best_candidate
from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.hybrid.electrostatics_repair import repair_electrostatics_schema_with_question




def _attach_formula_candidates(schema: dict[str, Any], prompt_result: PromptBuildResult) -> None:
    """Attach retrieval-ranked equations formulas for solver portfolio fallback.

    The prompt already shows these formula docs to the LLM, but the exact solver
    should also receive them explicitly so it can try nearby formulas when the
    LLM picked the wrong relation name.
    """
    if prompt_result.domain_decision.domain != "equations":
        return

    candidates: list[str] = []

    for formula in prompt_result.formulas:
        for key in ("id", "formula_id", "name", "formula"):
            value = formula.get(key)
            if isinstance(value, str) and value.strip():
                if value.strip() not in candidates:
                    candidates.append(value.strip())
                break

    for value in prompt_result.retrieval_debug.get("selected_formula_ids", []):
        if isinstance(value, str) and value.strip() and value.strip() not in candidates:
            candidates.append(value.strip())

    if not candidates:
        return

    existing = schema.get("formula_candidates")
    if isinstance(existing, str):
        existing_values = [existing]
    elif isinstance(existing, list):
        existing_values = [x for x in existing if isinstance(x, str)]
    else:
        existing_values = []

    merged: list[str] = []
    for value in existing_values + candidates:
        if value not in merged:
            merged.append(value)

    schema["formula_candidates"] = merged


def _equations_hybrid_candidates(
    problem: str,
    prompt_result: PromptBuildResult,
    *,
    llm_schema: dict[str, Any] | None = None,
    deterministic_schema: dict[str, Any] | None = None,
) -> list[SchemaCandidate]:
    """Build equations schema candidates from deterministic and LLM sources."""
    candidates: list[SchemaCandidate] = []

    if deterministic_schema is not None:
        _attach_formula_candidates(deterministic_schema, prompt_result)
        candidates.append(SchemaCandidate("deterministic_text", deterministic_schema))

    for schema in generate_equations_candidate_schemas(problem):
        _attach_formula_candidates(schema, prompt_result)
        candidates.append(SchemaCandidate("formula_driven", schema))

    if llm_schema is not None:
        _attach_formula_candidates(llm_schema, prompt_result)
        candidates.append(SchemaCandidate("llm_raw", llm_schema))

    return candidates


def _try_equations_hybrid_selection(
    problem: str,
    prompt_result: PromptBuildResult,
    *,
    raw_llm_output: str,
    llm_schema: dict[str, Any] | None = None,
    deterministic_schema: dict[str, Any] | None = None,
) -> SchemaPipelineResult | None:
    formula_candidates = generate_equations_candidate_schemas(problem)
    domain = prompt_result.domain_decision.domain
    if domain not in {"equations", "electrostatics"}:
        has_voltage_formula_candidate = any(
            any(obj.get("role") == "query" and obj.get("type") == "voltage" for obj in schema.get("objects", []))
            for schema in formula_candidates
        )
        if not has_voltage_formula_candidate:
            return None
    if domain != "equations" and not formula_candidates:
        return None

    candidates = _equations_hybrid_candidates(
        problem,
        prompt_result,
        llm_schema=llm_schema,
        deterministic_schema=deterministic_schema,
    )
    if not candidates:
        return None

    selected = select_best_candidate(problem, candidates)
    if selected is None:
        return None

    result = selected.solve_result
    if (
        raw_llm_output == "__deterministic_electrostatics_text_extractor__"
        and prompt_result.domain_decision.domain == "electrostatics"
        and result.status == "ok"
    ):
        result.status = "solved"
    result.add_step("Prompt built", f"Domain: {prompt_result.domain_decision.domain}")
    result.add_step(
        "Hybrid schema candidate selected",
        f"Selected {selected.candidate.source} with score {selected.score:.1f}.",
        candidate_source=selected.candidate.source,
        candidate_score=selected.score,
        candidate_diagnostics=selected.diagnostics,
    )
    return SchemaPipelineResult(
        solve_result=result,
        prompt_result=prompt_result,
        raw_llm_output=raw_llm_output,
        schema=selected.candidate.schema,
    )


@dataclass(frozen=True)
class SchemaPipelineResult:
    solve_result: SolveResult
    prompt_result: PromptBuildResult
    raw_llm_output: str
    schema: dict[str, Any] | None


def solve_problem_with_llm(
    problem: str,
    client: SchemaLLMClient,
    *,
    k: int = 3,
) -> SchemaPipelineResult:
    """
    End-to-end schema extraction pipeline:

        problem
        -> build_schema_prompt(problem)
        -> client.generate(prompt)
        -> extract_json_object(raw_output)
        -> schema_solver.solve_schema(schema)

    This function is model-agnostic. Tests should use a fake client.
    Real API clients should live behind SchemaLLMClient and be optional.
    """
    prompt_result = build_schema_prompt(problem, k=k)

    if prompt_result.domain_decision.domain == "electrostatics":
        schema = extract_electrostatics_schema_from_text(problem)
        if schema is not None:
            schema = repair_electrostatics_schema_with_question(problem, schema)

            # Some electrostatics questions are inverse/scalar formula problems
            # (for example, find the point where E=0).  The deterministic
            # geometry extractor may emit a checkable-but-wrong electric-field
            # query or an incomplete geometry schema.  Give formula-driven
            # candidates a chance before returning the deterministic result.
            hybrid = _try_equations_hybrid_selection(
                problem,
                prompt_result,
                raw_llm_output="__deterministic_electrostatics_text_extractor__",
                deterministic_schema=schema,
            )
            if hybrid is not None:
                return hybrid

            result = solve_schema(schema)
            result.add_step("Prompt built", f"Domain: {prompt_result.domain_decision.domain}")
            result.add_step("Deterministic electrostatics schema extracted", "Used narrow geometry/text extractor before LLM generation.")
            return SchemaPipelineResult(
                solve_result=result,
                prompt_result=prompt_result,
                raw_llm_output="__deterministic_electrostatics_text_extractor__",
                schema=schema,
            )

    raw_output = client.generate(prompt_result.prompt)

    deterministic_schema = extract_electrostatics_schema_from_text(problem)
    if deterministic_schema is None and prompt_result.domain_decision.domain == "equations":
        deterministic_schema = extract_equations_schema_from_text(problem)

    try:
        schema = extract_json_object(raw_output)
    except ValueError as exc:
        hybrid = _try_equations_hybrid_selection(
            problem,
            prompt_result,
            raw_llm_output=raw_output,
            deterministic_schema=deterministic_schema,
        )
        if hybrid is not None:
            return hybrid

        if deterministic_schema is not None:
            _attach_formula_candidates(deterministic_schema, prompt_result)
            result = solve_schema(deterministic_schema)
            result.add_step("Prompt built", f"Domain: {prompt_result.domain_decision.domain}")
            result.add_step(
                "Deterministic text repair",
                "LLM JSON parsing failed, so a canonical schema was extracted from the problem text.",
            )
            return SchemaPipelineResult(
                solve_result=result,
                prompt_result=prompt_result,
                raw_llm_output=raw_output,
                schema=deterministic_schema,
            )

        result = SolveResult(
            status="solve_failed",
            domain=prompt_result.domain_decision.domain,
            error=str(exc),
        )
        result.add_step("Prompt built", f"Domain: {prompt_result.domain_decision.domain}")
        result.add_step("LLM JSON extraction failed", str(exc))
        return SchemaPipelineResult(
            solve_result=result,
            prompt_result=prompt_result,
            raw_llm_output=raw_output,
            schema=None,
        )

    hybrid = _try_equations_hybrid_selection(
        problem,
        prompt_result,
        raw_llm_output=raw_output,
        llm_schema=schema,
        deterministic_schema=deterministic_schema,
    )
    if hybrid is not None:
        return hybrid

    # Electrostatics is especially sensitive to small schema mistakes: Qwen often
    # emits Collinear where the text gives AB/AC/BC, or encodes electric-field
    # targets as net_force targets. Prefer the deterministic text schema when it
    # recognizes a benchmark-safe pattern; otherwise fall back to the LLM schema.
    if deterministic_schema is not None:
        schema = deterministic_schema

    if prompt_result.domain_decision.domain == "electrostatics":
        schema = repair_electrostatics_schema_with_question(problem, schema)

    _attach_formula_candidates(schema, prompt_result)
    result = solve_schema(schema)
    result.add_step("Prompt built", f"Domain: {prompt_result.domain_decision.domain}")
    if deterministic_schema is not None:
        result.add_step(
            "Deterministic text repair",
            "A canonical schema was extracted from the problem text and used instead of the raw LLM schema.",
        )
    else:
        result.add_step("LLM schema extracted", "Schema JSON was parsed and sent to schema_solver.")

    return SchemaPipelineResult(
        solve_result=result,
        prompt_result=prompt_result,
        raw_llm_output=raw_output,
        schema=schema,
    )
