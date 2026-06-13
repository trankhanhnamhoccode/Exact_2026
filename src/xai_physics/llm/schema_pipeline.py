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

    # Electrostatics is especially sensitive to small schema mistakes: Qwen often
    # emits Collinear where the text gives AB/AC/BC, or encodes electric-field
    # targets as net_force targets. Prefer the deterministic text schema when it
    # recognizes a benchmark-safe pattern; otherwise fall back to the LLM schema.
    if deterministic_schema is not None:
        schema = deterministic_schema

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
