from xai_physics.domains.equations.retrieval.backends import EmbeddingBackend, RerankerBackend
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context
from xai_physics.domains.equations.retrieval.types import ExampleDoc, FormulaDoc, TagHit


def _formula_ids(result):
    return [item.formula.id for item in result.selected_formulas]


def _example_formula_ids(result):
    return [item.example.formula_id for item in result.selected_examples]


def test_retrieve_capacitor_energy_formula_and_example():
    result = retrieve_equations_context(
        "Calculate the energy stored in a capacitor with capacitance 100 uF and voltage 30 V.",
        formula_top_k=3,
        example_top_k=2,
    )

    assert _formula_ids(result)[0] == "capacitor_energy_voltage"
    assert "capacitor_energy_voltage" in _example_formula_ids(result)
    assert result.selected_formulas[0].formula.schema_template["relations"][0]["name"] == "capacitor_energy_voltage"


def test_retrieve_lc_resonance_formula():
    result = retrieve_equations_context(
        "Find the resonant frequency of an LC circuit with inductance 0.5 H and capacitance 5 uF.",
        formula_top_k=3,
        example_top_k=2,
    )

    assert _formula_ids(result)[0] == "lc_resonance_frequency"
    assert "lc_resonance_frequency" in _example_formula_ids(result)


def test_retrieve_solenoid_magnetic_field_formula():
    result = retrieve_equations_context(
        "A solenoid has 1000 turns, length 0.5 m, and current 2 A. Calculate the magnetic field.",
        formula_top_k=3,
        example_top_k=2,
    )

    assert _formula_ids(result)[0] == "solenoid_magnetic_field"
    assert "solenoid_magnetic_field" in _example_formula_ids(result)


def test_retrieve_point_charge_electric_field_formula():
    result = retrieve_equations_context(
        "Find the electric field at a distance of 30 cm from a point charge of 2 uC.",
        formula_top_k=3,
        example_top_k=2,
    )

    assert _formula_ids(result)[0] == "point_charge_electric_field"
    assert "point_charge_electric_field" in _example_formula_ids(result)


def test_retrieve_ohm_law_formula():
    result = retrieve_equations_context(
        "Find the current when voltage is 120 V and resistance is 40 ohm.",
        formula_top_k=3,
        example_top_k=2,
    )

    assert _formula_ids(result)[0] == "ohm_law"
    assert "ohm_law" in _example_formula_ids(result)


def test_retrieval_debug_contains_rule_and_tag_info():
    result = retrieve_equations_context(
        "Calculate the energy stored in a capacitor with capacitance 100 uF and voltage 30 V.",
        formula_top_k=3,
        example_top_k=2,
    )

    assert "formula_rule_scores" in result.debug
    assert "tag_hits" in result.debug
    assert "capacitor" in result.final_tags


def test_pipeline_accepts_custom_embedding_and_reranker_backends():
    class DummyEmbeddingBackend:
        def score(self, query: str, document: str) -> float:
            if "quality_factor" in document:
                return 99.0
            return 0.0

    class DummyRerankerBackend:
        def score_formula(
            self,
            query: str,
            formula: FormulaDoc,
            *,
            vector_score: float,
            rule_score: float,
            tag_score: float,
            matched_tags: list[str],
            tag_hits: list[TagHit],
        ) -> float:
            return vector_score

        def score_example(
            self,
            query: str,
            example: ExampleDoc,
            *,
            vector_score: float,
            formula_score: float,
            tag_score: float,
            matched_tags: list[str],
            tag_hits: list[TagHit],
        ) -> float:
            return vector_score + formula_score

    result = retrieve_equations_context(
        "dummy query",
        formula_top_k=1,
        example_top_k=1,
        embedding_backend=DummyEmbeddingBackend(),
        reranker_backend=DummyRerankerBackend(),
    )

    assert _formula_ids(result)[0] == "quality_factor"
