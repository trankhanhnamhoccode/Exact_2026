from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from xai_physics.llm.domain_classifier import classify_domain, DomainDecision
from xai_physics.llm.domain_prompts import DOMAIN_PROMPTS
from xai_physics.domains.capacitor_state.retrieval.pipeline import retrieve_capacitor_context
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context
from xai_physics.domains.electrostatics.retrieval.pipeline import retrieve_electrostatics_context


@dataclass
class PromptBuildResult:
    domain_decision: DomainDecision
    prompt: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    formulas: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    retrieval_debug: dict[str, Any] = field(default_factory=dict)


def format_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return ""

    blocks: list[str] = []

    for i, ex in enumerate(examples, 1):
        problem = ex.get("problem", "")
        schema = ex.get("schema", {})
        tags = ex.get("tags", [])

        schema_json = json.dumps(schema, ensure_ascii=False, indent=2)

        blocks.append(
            f"Example {i}\n"
            f"ID: {ex.get('id', '')}\n"
            f"Tags: {tags}\n"
            f"Problem:\n{problem}\n\n"
            f"JSON:\n{schema_json}\n"
        )

    return "\n".join(blocks)


def format_formula_docs(formulas: list[dict[str, Any]]) -> str:
    if not formulas:
        return ""

    blocks: list[str] = []

    for i, item in enumerate(formulas, 1):
        schema_template = item.get("schema_template", {})
        schema_json = json.dumps(schema_template, ensure_ascii=False, indent=2)

        blocks.append(
            f"Formula {i}\n"
            f"ID: {item.get('id', '')}\n"
            f"Name: {item.get('name', '')}\n"
            f"Equation: {item.get('equation', '')}\n"
            f"Description: {item.get('description', '')}\n"
            f"Quantity types: {item.get('quantity_types', [])}\n"
            f"Query types: {item.get('query_types', [])}\n"
            f"Tags: {item.get('tags', [])}\n"
            f"Schema template:\n{schema_json}\n"
        )

    return "\n".join(blocks)


def _build_capacitor_examples(problem: str, k: int) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    retrieval = retrieve_capacitor_context(
        problem=problem,
        vector_top_k=max(5, k + 2),
        final_top_k=k,
    )

    examples: list[dict[str, Any]] = []

    for item in retrieval.selected_examples:
        ex = item.example
        examples.append(
            {
                "id": ex.id,
                "domain": ex.domain,
                "tags": ex.tags,
                "problem": ex.problem,
                "schema": ex.schema,
                "retrieval": {
                    "vector_score": item.vector_score,
                    "rule_score": item.rule_score,
                    "rerank_score": item.rerank_score,
                    "matched_tags": item.matched_tags,
                },
            }
        )

    return examples, retrieval.final_tags, retrieval.debug


def _build_equations_context(
    problem: str,
    k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any]]:
    retrieval = retrieve_equations_context(
        problem=problem,
        formula_top_k=max(5, k + 3),
        example_top_k=max(5, k + 3),
    )

    formulas: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for item in retrieval.selected_formulas:
        formula = item.formula
        formulas.append(
            {
                "id": formula.id,
                "name": formula.name,
                "equation": formula.equation,
                "description": formula.description,
                "quantity_types": formula.quantity_types,
                "query_types": formula.query_types,
                "tags": formula.tags,
                "schema_template": formula.schema_template,
                "retrieval": {
                    "vector_score": item.vector_score,
                    "rule_score": item.rule_score,
                    "tag_score": item.tag_score,
                    "rerank_score": item.rerank_score,
                    "matched_tags": item.matched_tags,
                },
            }
        )

    for item in retrieval.selected_examples:
        ex = item.example
        examples.append(
            {
                "id": ex.id,
                "domain": "equations",
                "formula_id": ex.formula_id,
                "tags": ex.tags,
                "problem": ex.problem,
                "schema": ex.schema,
                "retrieval": {
                    "vector_score": item.vector_score,
                    "formula_score": item.formula_score,
                    "tag_score": item.tag_score,
                    "rerank_score": item.rerank_score,
                    "matched_tags": item.matched_tags,
                },
            }
        )

    return examples, formulas, retrieval.final_tags, retrieval.debug


def _build_electrostatics_examples(problem: str, k: int) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    retrieval = retrieve_electrostatics_context(
        problem=problem,
        vector_top_k=max(8, k + 4),
        final_top_k=max(4, k),
    )

    examples: list[dict[str, Any]] = []

    for item in retrieval.selected_examples:
        ex = item.example
        examples.append(
            {
                "id": ex.id,
                "domain": ex.domain,
                "tags": ex.tags,
                "problem": ex.problem,
                "schema": ex.schema,
                "retrieval": {
                    "vector_score": item.vector_score,
                    "rule_score": item.rule_score,
                    "rerank_score": item.rerank_score,
                    "matched_tags": item.matched_tags,
                },
            }
        )

    return examples, retrieval.final_tags, retrieval.debug


def build_schema_prompt(
    problem: str,
    examples: list[dict[str, Any]] | None = None,
    k: int = 3,
) -> PromptBuildResult:
    decision = classify_domain(problem)

    if decision.domain not in DOMAIN_PROMPTS:
        prompt = (
            "The problem could not be classified into a supported domain.\n"
            "Supported domains: capacitor_state, electrostatics, equations.\n"
            f"Problem:\n{problem}\n"
        )
        return PromptBuildResult(
            domain_decision=decision,
            prompt=prompt,
            examples=[],
            formulas=[],
            tags=decision.tags,
            retrieval_debug={},
        )

    retrieval_debug: dict[str, Any] = {}
    final_tags = list(decision.tags)
    formulas: list[dict[str, Any]] = []

    if examples is None:
        if decision.domain == "capacitor_state":
            examples, final_tags, retrieval_debug = _build_capacitor_examples(problem, k=k)
        elif decision.domain == "equations":
            examples, formulas, final_tags, retrieval_debug = _build_equations_context(problem, k=k)
        elif decision.domain == "electrostatics":
            examples, final_tags, retrieval_debug = _build_electrostatics_examples(problem, k=k)
        else:
            examples = []

    base_prompt = DOMAIN_PROMPTS[decision.domain]
    formula_block = format_formula_docs(formulas)
    example_block = format_examples(examples)

    prompt_parts = [
        base_prompt.strip(),
    ]

    if final_tags:
        prompt_parts.append("\nDetected tags:\n")
        prompt_parts.append(json.dumps(final_tags, ensure_ascii=False))

    if formula_block:
        prompt_parts.append("\nRelevant formula docs:\n")
        prompt_parts.append(formula_block)

    if example_block:
        prompt_parts.append("\nRetrieved examples:\n")
        prompt_parts.append(example_block)

    prompt_parts.append("\nUser problem:\n")
    prompt_parts.append(problem)
    prompt_parts.append("\nReturn JSON only.")

    return PromptBuildResult(
        domain_decision=decision,
        prompt="\n".join(prompt_parts),
        examples=examples,
        formulas=formulas,
        tags=final_tags,
        retrieval_debug=retrieval_debug,
    )
