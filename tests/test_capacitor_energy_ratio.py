from xai_physics.domains.capacitor_state.engine import solve_schema


def _solve_distance_energy_ratio(factor: float):
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 8, "unit": "uF"},
                "voltage": {"value": 100, "unit": "V"},
                "connected_to_source": True,
            }
        ],
        "events": [
            {"type": "DisconnectFromSource", "apply_to": ["C1"]},
            {"type": "DistanceScale", "apply_to": ["C1"], "params": {"factor": factor}},
        ],
        "queries": [
            {"type": "energy_ratio", "target": "C1", "unit": "times"}
        ],
    }
    return solve_schema(schema)


def test_energy_ratio_distance_doubled_disconnected():
    result = _solve_distance_energy_ratio(2)

    assert result.status == "solved"
    assert result.answer == "2 times"


def test_energy_ratio_distance_tripled_disconnected():
    result = _solve_distance_energy_ratio(3)

    assert result.status == "solved"
    assert result.answer == "3 times"


def test_energy_ratio_distance_quadrupled_disconnected():
    result = _solve_distance_energy_ratio(4)

    assert result.status == "solved"
    assert result.answer == "4 times"
