from xai_physics.domains.capacitor_state.engine import solve_schema


def test_replace_dielectric_capacitance_ratio_halves_when_k_4_to_2():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 1, "unit": "F"},
                "voltage": {"value": 1, "unit": "V"},
                "connected_to_source": False,
            }
        ],
        "events": [
            {
                "type": "ReplaceDielectric",
                "apply_to": ["C1"],
                "params": {"initial_k": 4, "final_k": 2},
            }
        ],
        "queries": [
            {"type": "capacitance_ratio", "target": "C1", "unit": "times"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "0.5 times"


def test_replace_dielectric_capacitance_ratio_doubles_when_k_2_to_4():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 1, "unit": "F"},
                "voltage": {"value": 1, "unit": "V"},
                "connected_to_source": False,
            }
        ],
        "events": [
            {
                "type": "ReplaceDielectric",
                "apply_to": ["C1"],
                "params": {"initial_k": 2, "final_k": 4},
            }
        ],
        "queries": [
            {"type": "capacitance_ratio", "target": "C1", "unit": "times"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "2 times"
