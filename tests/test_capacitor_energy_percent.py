from xai_physics.domains.capacitor_state.engine import solve_schema


def test_energy_percent_after_disconnected_dielectric_k2():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 10, "unit": "uF"},
                "voltage": {"value": 100, "unit": "V"},
                "connected_to_source": True,
            }
        ],
        "events": [
            {"type": "DisconnectFromSource", "apply_to": ["C1"]},
            {"type": "InsertDielectric", "apply_to": ["C1"], "params": {"k": 2}},
        ],
        "queries": [
            {"type": "energy_percent", "target": "C1", "unit": "%"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "50 %"


def test_energy_percent_after_disconnected_distance_doubled():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 10, "unit": "uF"},
                "voltage": {"value": 100, "unit": "V"},
                "connected_to_source": True,
            }
        ],
        "events": [
            {"type": "DisconnectFromSource", "apply_to": ["C1"]},
            {"type": "DistanceScale", "apply_to": ["C1"], "params": {"factor": 2}},
        ],
        "queries": [
            {"type": "energy_percent", "target": "C1", "unit": "%"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "200 %"
