from __future__ import annotations

from xai_physics.domains.capacitor_state.retrieval.example_store import load_examples
from xai_physics.domains.capacitor_state.retrieval.hard_rules import apply_hard_rules
from xai_physics.domains.capacitor_state.retrieval.reranker import rerank_examples
from xai_physics.domains.capacitor_state.retrieval.types import (
    CapacitorRetrievalResult,
    TagHit,
)
from xai_physics.domains.capacitor_state.retrieval.vector_store import VectorExampleIndex


def _soft_tags_from_examples(selected) -> list[TagHit]:
    tag_scores: dict[str, float] = {}

    for item in selected:
        for tag in item.example.tags:
            tag_scores[tag] = max(tag_scores.get(tag, 0.0), item.rerank_score)

    return [
        TagHit(
            tag=tag,
            source="retrieved_example",
            score=score,
            evidence="Tag appeared in a selected retrieved example.",
        )
        for tag, score in sorted(tag_scores.items(), key=lambda kv: kv[1], reverse=True)
    ]


def retrieve_capacitor_context(
    problem: str,
    vector_top_k: int = 5,
    final_top_k: int = 3,
) -> CapacitorRetrievalResult:
    examples = load_examples()

    hard_tags = apply_hard_rules(problem)
    hard_tag_names = [hit.tag for hit in hard_tags]

    index = VectorExampleIndex(examples)
    vector_results = index.search(problem, top_k=vector_top_k)

    selected = rerank_examples(
        vector_results=vector_results,
        hard_tags=hard_tags,
        top_k=final_top_k,
    )

    soft_tags = _soft_tags_from_examples(selected)

    final_tags: list[str] = []
    for tag in hard_tag_names:
        if tag not in final_tags:
            final_tags.append(tag)

    for hit in soft_tags:
        if hit.tag not in final_tags:
            final_tags.append(hit.tag)

    return CapacitorRetrievalResult(
        domain="capacitor_state",
        hard_tags=hard_tags,
        soft_tags=soft_tags,
        final_tags=final_tags,
        selected_examples=selected,
        debug={
            "vector_top_k": vector_top_k,
            "final_top_k": final_top_k,
            "vector_candidates": [
                {
                    "id": ex.id,
                    "score": score,
                    "tags": ex.tags,
                }
                for ex, score in vector_results
            ],
        },
    )
