from __future__ import annotations

from xai_physics.domains.electrostatics.retrieval.types import RetrievedExample, SchemaExample, TagHit


IMPORTANT_TAGS = {
    "resultant_vector",
    "electric_field",
    "pairwise_distances",
    "perpendicular_bisector",
    "midpoint",
    "isosceles_right",
    "foot_of_altitude",
    "perpendicular_fields",
}


def rerank_examples(
    vector_results: list[tuple[SchemaExample, float]],
    hard_tags: list[TagHit],
    top_k: int = 4,
) -> list[RetrievedExample]:
    hard_tag_set = {hit.tag for hit in hard_tags}
    ranked: list[RetrievedExample] = []

    for ex, vector_score in vector_results:
        matched = sorted(set(ex.tags) & hard_tag_set)
        rule_score = 2.0 * len(matched)
        rule_score += sum(2.0 for tag in matched if tag in IMPORTANT_TAGS)
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
