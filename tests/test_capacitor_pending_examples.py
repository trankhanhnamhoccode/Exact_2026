from xai_physics.domains.capacitor_state.retrieval.pending_store import (
    load_pending_examples,
    filter_pending_by_tag,
    filter_pending_by_feature,
)


def test_pending_examples_load():
    examples = load_pending_examples()

    assert len(examples) >= 5
    assert all(ex.domain == "capacitor_state" for ex in examples)
    assert all(ex.status == "pending" for ex in examples)


def test_short_circuit_is_no_longer_pending():
    examples = filter_pending_by_tag("short_circuit")

    assert examples == []


def test_distance_energy_scaling_is_no_longer_pending_but_nl311_remains():
    symbolic = filter_pending_by_feature("symbolic scaling")

    assert any(ex.id == "cap_pending_nl311" for ex in symbolic)
    assert all(ex.id not in {
        "cap_pending_nl323",
        "cap_pending_nl328",
        "cap_pending_nl335",
        "cap_pending_nl378",
    } for ex in symbolic)


def test_lc_oscillation_is_no_longer_pending():
    examples = filter_pending_by_tag("inductor_oscillation")

    assert examples == []
