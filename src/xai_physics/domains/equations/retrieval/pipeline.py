from __future__ import annotations

from .backends import EmbeddingBackend, RerankerBackend
from .example_store import load_examples
from .formula_store import list_formula_docs
from .hard_rules import detect_tags, formula_rule_scores
from .reranker import rank_examples, rank_formulas
from .types import EquationRetrievalResult


def retrieve_equations_context(
    problem: str,
    formula_top_k: int = 5,
    example_top_k: int = 3,
    embedding_backend: EmbeddingBackend | None = None,
    reranker_backend: RerankerBackend | None = None,
) -> EquationRetrievalResult:
    tag_hits = detect_tags(problem)
    final_tags = [hit.tag for hit in tag_hits]

    formulas = list_formula_docs()
    selected_formulas = rank_formulas(
        problem=problem,
        formulas=formulas,
        tag_hits=tag_hits,
        top_k=formula_top_k,
        embedding_backend=embedding_backend,
        reranker_backend=reranker_backend,
    )

    selected_formula_ids = [item.formula.id for item in selected_formulas]

    examples = load_examples()
    selected_examples = rank_examples(
        problem=problem,
        examples=examples,
        selected_formula_ids=selected_formula_ids,
        tag_hits=tag_hits,
        top_k=example_top_k,
        embedding_backend=embedding_backend,
        reranker_backend=reranker_backend,
    )

    return EquationRetrievalResult(
        selected_formulas=selected_formulas,
        selected_examples=selected_examples,
        final_tags=final_tags,
        debug={
            "formula_rule_scores": formula_rule_scores(problem),
            "tag_hits": [
                {
                    "tag": hit.tag,
                    "score": hit.score,
                    "evidence": hit.evidence,
                }
                for hit in tag_hits
            ],
            "selected_formula_ids": selected_formula_ids,
            "selected_example_ids": [item.example.id for item in selected_examples],
        },
    )
