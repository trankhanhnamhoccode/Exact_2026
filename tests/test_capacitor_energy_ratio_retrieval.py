from xai_physics.domains.capacitor_state.retrieval.pipeline import retrieve_capacitor_context
from xai_physics.llm.prompt_builder import build_schema_prompt


def test_distance_energy_ratio_retrieval_selects_ratio_example():
    problem = (
        "A parallel-plate capacitor is charged and disconnected from the source. "
        "If the distance between the plates is tripled, how will the electric field energy change?"
    )

    result = retrieve_capacitor_context(problem)

    assert "distance_scale" in result.final_tags
    assert "ratio_query" in result.final_tags
    assert result.selected_examples
    assert result.selected_examples[0].example.id in {
        "cap_nl323_distance_energy_ratio",
        "cap_nl328_distance_energy_ratio",
        "cap_nl335_distance_energy_ratio",
        "cap_nl378_distance_energy_ratio",
    }


def test_distance_energy_ratio_prompt_contains_energy_ratio():
    problem = (
        "A capacitor is disconnected from the source. "
        "If the distance between the plates is quadrupled, how many times the initial energy is the new energy?"
    )

    built = build_schema_prompt(problem)

    assert built.domain_decision.domain == "capacitor_state"
    assert "energy_ratio" in built.prompt
    assert built.examples
