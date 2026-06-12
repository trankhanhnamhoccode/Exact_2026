from xai_physics.domains.capacitor_state.retrieval.example_store import load_examples
from xai_physics.domains.capacitor_state.retrieval.hard_rules import apply_hard_rules
from xai_physics.domains.capacitor_state.retrieval.pipeline import retrieve_capacitor_context


def test_capacitor_examples_load():
    examples = load_examples()

    assert len(examples) >= 10
    assert all(ex.domain == "capacitor_state" for ex in examples)


def test_hard_rules_detect_isolated_dielectric():
    problem = (
        "A capacitor is disconnected from the battery and a dielectric "
        "with constant 2 is inserted. Find the final voltage."
    )

    hits = apply_hard_rules(problem)
    tags = {hit.tag for hit in hits}

    assert "dielectric" in tags
    assert "disconnect" in tags
    assert "isolated_dielectric" in tags
    assert "voltage_query" in tags


def test_hard_rules_detect_parallel_redistribution():
    problem = (
        "Two capacitors are charged separately and then connected with "
        "their like-poled terminals together. Find the final voltage."
    )

    hits = apply_hard_rules(problem)
    tags = {hit.tag for hit in hits}

    assert "parallel_connection" in tags or "like_polarity_connection" in tags


def test_retrieve_capacitor_context_dielectric_prefers_dielectric_example():
    problem = (
        "A capacitor is disconnected from a source. Then a dielectric is inserted. "
        "Find the final voltage."
    )

    result = retrieve_capacitor_context(problem)

    assert "isolated_dielectric" in result.final_tags
    assert result.selected_examples
    assert result.selected_examples[0].example.id == "cap_td001"


def test_retrieve_capacitor_context_parallel_prefers_parallel_example():
    problem = (
        "Two charged capacitors are connected with their like-polarity terminals together. "
        "Find the final voltage."
    )

    result = retrieve_capacitor_context(problem)

    assert result.selected_examples
    assert result.selected_examples[0].example.id in {"cap_td095", "cap_td381"}
