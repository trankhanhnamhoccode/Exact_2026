from xai_physics.domains.capacitor_state.retrieval.pipeline import retrieve_capacitor_context
from xai_physics.llm.prompt_builder import build_schema_prompt


def test_replace_dielectric_retrieval_selects_td386():
    problem = (
        "A capacitor is filled with a dielectric of dielectric constant 4. "
        "The dielectric is replaced by another material with dielectric constant 2. "
        "What is the ratio of the new capacitance to the initial capacitance?"
    )

    result = retrieve_capacitor_context(problem)

    assert "replace_dielectric" in result.final_tags
    assert "capacitance_ratio_query" in result.final_tags
    assert result.selected_examples
    assert result.selected_examples[0].example.id == "cap_td386_replace_dielectric_capacitance_ratio"


def test_replace_dielectric_prompt_contains_schema_terms():
    problem = (
        "A capacitor has dielectric constant 4 and it is replaced by dielectric constant 2. "
        "Find the ratio of final capacitance to initial capacitance."
    )

    built = build_schema_prompt(problem)

    assert built.domain_decision.domain == "capacitor_state"
    assert "ReplaceDielectric" in built.prompt
    assert "capacitance_ratio" in built.prompt
