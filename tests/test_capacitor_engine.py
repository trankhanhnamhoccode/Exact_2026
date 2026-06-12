from xai_physics.domains.capacitor_state.engine import solve_schema


def test_engine_single_capacitor_disconnected_dielectric():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 500, "unit": "pF"},
                "voltage": {"value": 300, "unit": "V"},
                "connected_to_source": True,
            }
        ],
        "events": [
            {
                "type": "DisconnectFromSource",
                "apply_to": ["C1"],
            },
            {
                "type": "InsertDielectric",
                "apply_to": ["C1"],
                "params": {
                    "dielectric_constant": 2
                },
            },
        ],
        "queries": [
            {
                "type": "voltage",
                "target": "C1",
                "unit": "V",
            }
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "150 V"


def test_engine_parallel_redistribution_final_voltage():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 2, "unit": "uF"},
                "voltage": {"value": 100, "unit": "V"},
                "connected_to_source": False,
            },
            {
                "id": "C2",
                "type": "capacitor",
                "capacitance": {"value": 3, "unit": "uF"},
                "voltage": {"value": 0, "unit": "V"},
                "connected_to_source": False,
            },
        ],
        "events": [
            {
                "type": "ParallelRedistribution",
                "apply_to": ["C1", "C2"],
                "params": {
                    "polarity": "same"
                },
            }
        ],
        "queries": [
            {
                "type": "voltage",
                "target": "system",
                "unit": "V",
            }
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "40 V"


def test_engine_parallel_redistribution_total_charge():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 2, "unit": "uF"},
                "voltage": {"value": 100, "unit": "V"},
                "connected_to_source": False,
            },
            {
                "id": "C2",
                "type": "capacitor",
                "capacitance": {"value": 3, "unit": "uF"},
                "voltage": {"value": 0, "unit": "V"},
                "connected_to_source": False,
            },
        ],
        "events": [
            {
                "type": "ParallelRedistribution",
                "apply_to": ["C1", "C2"],
                "params": {
                    "polarity": "same"
                },
            }
        ],
        "queries": [
            {
                "type": "charge",
                "target": "system",
                "unit": "uC",
            }
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "200 uC"


def test_engine_rejects_opposite_polarity_for_now():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 2, "unit": "uF"},
                "voltage": {"value": 100, "unit": "V"},
            },
            {
                "id": "C2",
                "type": "capacitor",
                "capacitance": {"value": 3, "unit": "uF"},
                "voltage": {"value": 0, "unit": "V"},
            },
        ],
        "events": [
            {
                "type": "ParallelRedistribution",
                "apply_to": ["C1", "C2"],
                "params": {
                    "polarity": "opposite"
                },
            }
        ],
        "queries": [
            {
                "type": "voltage",
                "target": "system",
                "unit": "V",
            }
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solve_failed"
    assert "same-polarity" in result.error
