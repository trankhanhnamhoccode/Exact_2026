from __future__ import annotations

from xai_physics.domains.electrostatics.retrieval.example_store import load_examples
from xai_physics.domains.electrostatics.retrieval.hard_rules import apply_hard_rules
from xai_physics.domains.electrostatics.retrieval.reranker import rerank_examples
from xai_physics.domains.electrostatics.retrieval.types import ElectrostaticsRetrievalResult
from xai_physics.domains.electrostatics.retrieval.vector_store import VectorExampleIndex


def retrieve_electrostatics_context(
    problem: str,
    vector_top_k: int = 8,
    final_top_k: int = 4,
) -> ElectrostaticsRetrievalResult:
    examples = load_examples()
    hard_tags = apply_hard_rules(problem)
    index = VectorExampleIndex(examples)
    vector_results = index.search(problem, top_k=vector_top_k)
    selected = rerank_examples(vector_results, hard_tags=hard_tags, top_k=final_top_k)

    final_tags: list[str] = []
    for hit in hard_tags:
        if hit.tag not in final_tags:
            final_tags.append(hit.tag)
    for item in selected:
        for tag in item.example.tags:
            if tag not in final_tags:
                final_tags.append(tag)

    return ElectrostaticsRetrievalResult(
        domain="electrostatics",
        hard_tags=hard_tags,
        final_tags=final_tags,
        selected_examples=selected,
        debug={
            "vector_top_k": vector_top_k,
            "final_top_k": final_top_k,
            "tag_hits": [hit.__dict__ for hit in hard_tags],
            "vector_candidates": [
                {"id": ex.id, "score": score, "tags": ex.tags}
                for ex, score in vector_results
            ],
            "selected_example_ids": [item.example.id for item in selected],
        },
    )
