from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from xai_physics.llm.domain_classifier import classify_domain, DomainDecision
from xai_physics.llm.domain_prompts import DOMAIN_PROMPTS
from xai_physics.domains.capacitor_state.retrieval.pipeline import retrieve_capacitor_context


@dataclass
class PromptBuildResult:
    domain_decision: DomainDecision
    prompt: str
    examples: list[dict[str, Any]] = field(default_factory=list)
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
            f"Tags: {tags}\n"
            f"Problem:\n{problem}\n\n"
            f"JSON:\n{schema_json}\n"
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


def build_schema_prompt(
    problem: str,
    examples: list[dict[str, Any]] | None = None,
    k: int = 3,
) -> PromptBuildResult:
    decision = classify_domain(problem)

    if decision.domain not in DOMAIN_PROMPTS:
        prompt = (
            "The problem could not be classified into a supported domain.\n"
            "Supported domains: capacitor_state, electrostatics.\n"
            f"Problem:\n{problem}\n"
        )
        return PromptBuildResult(
            domain_decision=decision,
            prompt=prompt,
            examples=[],
            tags=decision.tags,
            retrieval_debug={},
        )

    retrieval_debug: dict[str, Any] = {}
    final_tags = list(decision.tags)

    if examples is None:
        if decision.domain == "capacitor_state":
            examples, final_tags, retrieval_debug = _build_capacitor_examples(problem, k=k)
        else:
            examples = []

    base_prompt = DOMAIN_PROMPTS[decision.domain]
    example_block = format_examples(examples)

    prompt_parts = [
        base_prompt.strip(),
    ]

    if final_tags:
        prompt_parts.append("\nDetected tags:\n")
        prompt_parts.append(json.dumps(final_tags, ensure_ascii=False))

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
        tags=final_tags,
        retrieval_debug=retrieval_debug,
    )
