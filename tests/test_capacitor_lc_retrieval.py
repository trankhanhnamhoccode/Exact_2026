from xai_physics.llm.prompt_builder import build_schema_prompt
from xai_physics.domains.capacitor_state.retrieval.pipeline import retrieve_capacitor_context


def test_lc_oscillation_retrieval_selects_lc_example():
    problem = (
        "A capacitor C = 25 uF is charged at 120 V. After that, it is "
        "connected to an inductor. Calculate the total oscillation energy."
    )

    result = retrieve_capacitor_context(problem)

    assert "inductor_oscillation" in result.final_tags
    assert result.selected_examples
    assert result.selected_examples[0].example.id in {
        "cap_nl024_lc_energy",
        "cap_nl094_lc_energy",
    }


def test_lc_oscillation_prompt_contains_connect_to_inductor():
    problem = (
        "A capacitor C = 25 uF is charged at 120 V. After that, it is "
        "connected to an inductor. Calculate the total oscillation energy."
    )

    built = build_schema_prompt(problem)

    assert built.domain_decision.domain == "capacitor_state"
    assert "inductor_oscillation" in built.tags
    assert "ConnectToInductor" in built.prompt
    assert built.examples
