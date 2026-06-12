from xai_physics.domains.capacitor_state.engine import solve_schema


def test_short_circuit_charge_becomes_zero():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 3, "unit": "μF"},
                "voltage": {"value": 12, "unit": "V"},
                "connected_to_source": False,
            }
        ],
        "events": [
            {"type": "ShortCircuit", "apply_to": ["C1"]}
        ],
        "queries": [
            {"type": "charge", "target": "C1", "unit": "μC"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "0 μC"


def test_short_circuit_energy_becomes_zero():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 3, "unit": "μF"},
                "voltage": {"value": 12, "unit": "V"},
                "connected_to_source": False,
            }
        ],
        "events": [
            {"type": "ShortCircuit", "apply_to": ["C1"]}
        ],
        "queries": [
            {"type": "energy", "target": "C1", "unit": "μJ"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "0 μJ"


def test_short_circuit_voltage_becomes_zero():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 3, "unit": "μF"},
                "voltage": {"value": 12, "unit": "V"},
                "connected_to_source": False,
            }
        ],
        "events": [
            {"type": "ShortCircuit", "apply_to": ["C1"]}
        ],
        "queries": [
            {"type": "voltage", "target": "C1", "unit": "V"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "0 V"
