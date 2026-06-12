from xai_physics.domains.capacitor_state.retrieval.pending_store import (
    load_pending_examples,
    filter_pending_by_tag,
    filter_pending_by_feature,
)


def test_pending_examples_load():
    examples = load_pending_examples()

    assert len(examples) >= 10
    assert all(ex.domain == "capacitor_state" for ex in examples)
    assert all(ex.status == "pending" for ex in examples)


def test_pending_examples_include_short_circuit():
    examples = filter_pending_by_tag("short_circuit")

    assert examples
    assert examples[0].id == "cap_pending_td374"


def test_pending_examples_include_symbolic_scaling():
    examples = filter_pending_by_feature("symbolic scaling")

    assert len(examples) >= 4
    assert any(ex.id == "cap_pending_nl335" for ex in examples)


def test_pending_examples_include_lc_oscillation():
    examples = filter_pending_by_tag("inductor_oscillation")

    assert len(examples) >= 2
