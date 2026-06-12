from xai_physics.domains.capacitor_state.engine import solve_schema


def test_connect_to_inductor_energy_from_capacitor():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 25, "unit": "μF"},
                "voltage": {"value": 120, "unit": "V"},
                "connected_to_source": False,
            }
        ],
        "events": [
            {"type": "ConnectToInductor", "apply_to": ["C1"]}
        ],
        "queries": [
            {"type": "energy", "target": "system", "unit": "mJ"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "180 mJ"


def test_connect_to_inductor_energy_second_case():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 50, "unit": "μF"},
                "voltage": {"value": 100, "unit": "V"},
                "connected_to_source": False,
            }
        ],
        "events": [
            {"type": "ConnectToInductor", "apply_to": ["C1"]}
        ],
        "queries": [
            {"type": "energy", "target": "system", "unit": "mJ"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "250 mJ"
