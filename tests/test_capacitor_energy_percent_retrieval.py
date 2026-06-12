from xai_physics.domains.capacitor_state.retrieval.pipeline import retrieve_capacitor_context
from xai_physics.llm.prompt_builder import build_schema_prompt


def test_energy_percent_retrieval_selects_nl091():
    problem = (
        "A charged capacitor is disconnected from the source. "
        "A dielectric with dielectric constant 2 is inserted. "
        "What percentage of the initial energy remains?"
    )

    result = retrieve_capacitor_context(problem)

    assert "energy_percent_query" in result.final_tags
    assert "ratio_query" in result.final_tags
    assert result.selected_examples
    assert result.selected_examples[0].example.id == "cap_nl091_energy_percent_disconnected_dielectric"


def test_energy_percent_prompt_contains_energy_percent():
    problem = (
        "A capacitor is disconnected from the source and a dielectric is inserted. "
        "What percentage of the initial energy remains?"
    )

    built = build_schema_prompt(problem)

    assert built.domain_decision.domain == "capacitor_state"
    assert "energy_percent" in built.prompt
    assert built.examples
