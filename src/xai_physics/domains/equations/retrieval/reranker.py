from __future__ import annotations

from .backends import EmbeddingBackend, HeuristicRerankerBackend, LexicalEmbeddingBackend, RerankerBackend
from .hard_rules import formula_rule_scores
from .types import ExampleDoc, FormulaDoc, RetrievedExample, RetrievedFormula, TagHit


def _tag_score(doc_tags: list[str], tag_hits: list[TagHit]) -> tuple[float, list[str]]:
    hit_scores = {hit.tag: hit.score for hit in tag_hits}
    matched = [tag for tag in doc_tags if tag in hit_scores]
    score = sum(hit_scores[tag] for tag in matched)
    return score, matched


def rank_formulas(
    problem: str,
    formulas: list[FormulaDoc],
    tag_hits: list[TagHit],
    top_k: int,
    embedding_backend: EmbeddingBackend | None = None,
    reranker_backend: RerankerBackend | None = None,
) -> list[RetrievedFormula]:
    embedding_backend = embedding_backend or LexicalEmbeddingBackend()
    reranker_backend = reranker_backend or HeuristicRerankerBackend()

    rule_scores = formula_rule_scores(problem)
    ranked: list[RetrievedFormula] = []

    for formula in formulas:
        vector_score = embedding_backend.score(problem, formula.searchable_text())
        rule_score = rule_scores.get(formula.id, 0.0)
        tag_score, matched_tags = _tag_score(formula.tags, tag_hits)

        rerank_score = reranker_backend.score_formula(
            problem,
            formula,
            vector_score=vector_score,
            rule_score=rule_score,
            tag_score=tag_score,
            matched_tags=matched_tags,
            tag_hits=tag_hits,
        )

        ranked.append(
            RetrievedFormula(
                formula=formula,
                vector_score=vector_score,
                rule_score=rule_score,
                tag_score=tag_score,
                rerank_score=rerank_score,
                matched_tags=matched_tags,
            )
        )

    ranked.sort(key=lambda item: item.rerank_score, reverse=True)
    return ranked[:top_k]


def rank_examples(
    problem: str,
    examples: list[ExampleDoc],
    selected_formula_ids: list[str],
    tag_hits: list[TagHit],
    top_k: int,
    embedding_backend: EmbeddingBackend | None = None,
    reranker_backend: RerankerBackend | None = None,
) -> list[RetrievedExample]:
    embedding_backend = embedding_backend or LexicalEmbeddingBackend()
    reranker_backend = reranker_backend or HeuristicRerankerBackend()

    ranked: list[RetrievedExample] = []

    for example in examples:
        vector_score = embedding_backend.score(problem, example.searchable_text())
        formula_score = 2.0 if example.formula_id in selected_formula_ids else 0.0
        tag_score, matched_tags = _tag_score(example.tags, tag_hits)

        rerank_score = reranker_backend.score_example(
            problem,
            example,
            vector_score=vector_score,
            formula_score=formula_score,
            tag_score=tag_score,
            matched_tags=matched_tags,
            tag_hits=tag_hits,
        )

        ranked.append(
            RetrievedExample(
                example=example,
                vector_score=vector_score,
                formula_score=formula_score,
                tag_score=tag_score,
                rerank_score=rerank_score,
                matched_tags=matched_tags,
            )
        )

    ranked.sort(key=lambda item: item.rerank_score, reverse=True)
    return ranked[:top_k]
