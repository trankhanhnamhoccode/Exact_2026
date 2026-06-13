from __future__ import annotations

from xai_physics.domains.capacitor_state.retrieval.types import (
    RetrievedExample,
    SchemaExample,
    TagHit,
)


IMPORTANT_TAGS = {
    "isolated_dielectric",
    "source_dielectric",
    "charge_redistribution",
    "distance_scale",
    "area_scale",
    "replace_capacitor",
    "energy_reduction_query",
    "constant_voltage",
}


def rerank_examples(
    vector_results: list[tuple[SchemaExample, float]],
    hard_tags: list[TagHit],
    top_k: int = 3,
) -> list[RetrievedExample]:
    hard_tag_set = {hit.tag for hit in hard_tags}

    ranked: list[RetrievedExample] = []

    for ex, vector_score in vector_results:
        ex_tags = set(ex.tags)
        matched = sorted(ex_tags & hard_tag_set)

        rule_score = 0.0
        rule_score += 2.0 * len(matched)

        for tag in matched:
            if tag in IMPORTANT_TAGS:
                rule_score += 2.0

        # Final score is deliberately rule-heavy.
        # Hard rules should dominate when they catch important structure.
        final = vector_score + rule_score

        ranked.append(
            RetrievedExample(
                example=ex,
                vector_score=vector_score,
                rule_score=rule_score,
                rerank_score=final,
                matched_tags=matched,
            )
        )

    ranked.sort(key=lambda item: item.rerank_score, reverse=True)
    return ranked[:top_k]
