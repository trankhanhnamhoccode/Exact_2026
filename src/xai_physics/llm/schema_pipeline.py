from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xai_physics.core.result import SolveResult
from xai_physics.llm.client import SchemaLLMClient
from xai_physics.llm.json_extractor import extract_json_object
from xai_physics.llm.prompt_builder import PromptBuildResult, build_schema_prompt
from xai_physics.schema_solver import solve_schema


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
    raw_output = client.generate(prompt_result.prompt)

    try:
        schema = extract_json_object(raw_output)
    except ValueError as exc:
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

    result = solve_schema(schema)
    result.add_step("Prompt built", f"Domain: {prompt_result.domain_decision.domain}")
    result.add_step("LLM schema extracted", "Schema JSON was parsed and sent to schema_solver.")

    return SchemaPipelineResult(
        solve_result=result,
        prompt_result=prompt_result,
        raw_llm_output=raw_output,
        schema=schema,
    )
