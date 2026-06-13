import pytest

from xai_physics.domains.capacitor_state.engine import solve_schema
from xai_physics.domains.capacitor_state.retrieval.pipeline import retrieve_capacitor_context


def _num(answer: str) -> float:
    return float(answer.split()[0])


def test_source_work_when_distance_doubles_while_connected_to_source():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "area": {"value": 100, "unit": "cm^2"},
                "distance": {"value": 1, "unit": "mm"},
                "voltage": {"value": 200, "unit": "V"},
                "connected_to_source": True,
            }
        ],
        "events": [
            {"type": "DistanceScale", "apply_to": ["C1"], "params": {"factor": 2}},
        ],
        "queries": [{"type": "source_work", "target": "C1", "unit": "uJ"}],
    }
    r = solve_schema(schema)
    assert r.status == "solved", r.error
    assert _num(r.answer) == pytest.approx(-1.77084, rel=1e-6)


def test_source_work_query_aliases_work_by_source():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 4, "unit": "uF"},
                "voltage": {"value": 10, "unit": "V"},
                "connected_to_source": True,
            }
        ],
        "events": [
            {"type": "CapacitanceScale", "apply_to": ["C1"], "params": {"factor": 2, "hold": "voltage"}},
        ],
        "queries": [{"type": "work_by_source", "target": "C1", "unit": "uJ"}],
    }
    r = solve_schema(schema)
    assert r.status == "solved", r.error
    assert _num(r.answer) == pytest.approx(400.0)


def test_capacitance_scale_constant_voltage_energy_ratio():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 1, "unit": "F"},
                "voltage": {"value": 1, "unit": "V"},
                "connected_to_source": True,
            }
        ],
        "events": [
            {"type": "CapacitanceScale", "apply_to": ["C1"], "params": {"factor": 2, "hold": "voltage"}},
        ],
        "queries": [{"type": "energy_ratio", "target": "C1", "unit": "times"}],
    }
    r = solve_schema(schema)
    assert r.status == "solved", r.error
    assert _num(r.answer) == pytest.approx(2.0)


def test_capacitance_scale_disconnected_conserves_charge_voltage_changes():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 4, "unit": "uF"},
                "voltage": {"value": 12, "unit": "V"},
                "connected_to_source": False,
            }
        ],
        "events": [
            {"type": "CapacitanceScale", "apply_to": ["C1"], "params": {"factor": 2}},
        ],
        "queries": [{"type": "voltage", "target": "C1", "unit": "V"}],
    }
    r = solve_schema(schema)
    assert r.status == "solved", r.error
    assert _num(r.answer) == pytest.approx(6.0)


def test_retrieval_prioritizes_source_work_pending_td391_example():
    q = "A parallel plate capacitor has a plate separation d = 1 mm and is charged to U = 200 V. The plate area is S = 100 cm^2. The plate separation is then doubled while still connected to the source. Calculate the additional work supplied by the source."
    ctx = retrieve_capacitor_context(q, final_top_k=5)
    ids = [item.example.id for item in ctx.selected_examples]
    assert "cap_td391_source_work_distance_doubled" in ids


def test_retrieval_prioritizes_constant_voltage_capacitance_scale_example():
    q = "How does the energy stored in a capacitor change if the capacitance is doubled and the voltage is kept constant?"
    ctx = retrieve_capacitor_context(q, final_top_k=5)
    ids = [item.example.id for item in ctx.selected_examples]
    assert "cap_nl311_constant_voltage_capacitance_doubled_energy_ratio" in ids
