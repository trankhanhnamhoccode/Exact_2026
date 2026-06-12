from xai_physics.llm.prompt_builder import build_schema_prompt
from xai_physics.domains.capacitor_state.retrieval.pipeline import retrieve_capacitor_context


def test_short_circuit_retrieval_selects_td374_example():
    problem = (
        "A capacitor C = 3 uF is charged at 12 V, then its two plates are "
        "short-circuited. Calculate the energy after short-circuiting."
    )

    result = retrieve_capacitor_context(problem)

    assert "short_circuit" in result.final_tags
    assert result.selected_examples
    assert result.selected_examples[0].example.id in {
        "cap_td374_energy",
        "cap_td374_charge",
    }


def test_short_circuit_prompt_contains_short_circuit_event():
    problem = (
        "A capacitor C = 3 uF is charged at 12 V, then its two plates are "
        "short-circuited. Calculate the energy after short-circuiting."
    )

    built = build_schema_prompt(problem)

    assert built.domain_decision.domain == "capacitor_state"
    assert "short_circuit" in built.tags
    assert "ShortCircuit" in built.prompt
    assert built.examples
