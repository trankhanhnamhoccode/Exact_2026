from __future__ import annotations

from dataclasses import dataclass

from xai_physics.llm.domain_classifier import classify_domain, DomainDecision
from xai_physics.llm.domain_prompts import DOMAIN_PROMPTS


@dataclass
class PromptBuildResult:
    domain_decision: DomainDecision
    prompt: str


def format_examples(examples: list[dict]) -> str:
    if not examples:
        return ""

    blocks: list[str] = []

    for i, ex in enumerate(examples, 1):
        problem = ex.get("problem", "")
        schema = ex.get("schema", {})
        blocks.append(
            f"Example {i}\n"
            f"Problem:\n{problem}\n\n"
            f"JSON:\n{schema}\n"
        )

    return "\n".join(blocks)


def build_schema_prompt(problem: str, examples: list[dict] | None = None) -> PromptBuildResult:
    examples = examples or []

    decision = classify_domain(problem)

    if decision.domain not in DOMAIN_PROMPTS:
        prompt = (
            "The problem could not be classified into a supported domain.\n"
            "Supported domains: capacitor_state, electrostatics.\n"
            f"Problem:\n{problem}\n"
        )
        return PromptBuildResult(domain_decision=decision, prompt=prompt)

    base_prompt = DOMAIN_PROMPTS[decision.domain]
    example_block = format_examples(examples)

    prompt_parts = [
        base_prompt.strip(),
    ]

    if example_block:
        prompt_parts.append("\nRetrieved examples:\n")
        prompt_parts.append(example_block)

    prompt_parts.append("\nUser problem:\n")
    prompt_parts.append(problem)
    prompt_parts.append("\nReturn JSON only.")

    return PromptBuildResult(
        domain_decision=decision,
        prompt="\n".join(prompt_parts),
    )
