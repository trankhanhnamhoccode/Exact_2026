from xai_physics.schema_solver import solve_schema


def test_unified_solver_routes_capacitor_state():
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
            {"type": "DisconnectFromSource", "apply_to": ["C1"]},
            {
                "type": "InsertDielectric",
                "apply_to": ["C1"],
                "params": {"dielectric_constant": 2},
            },
        ],
        "queries": [
            {"type": "voltage", "target": "C1", "unit": "V"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.domain == "capacitor_state"
    assert result.answer == "150 V"


def test_unified_solver_routes_electrostatics():
    schema = {
        "domain": "electrostatics",
        "points": [
            {
                "id": "A",
                "x": {"value": 0, "unit": "m"},
                "y": {"value": 0, "unit": "m"},
            },
            {
                "id": "B",
                "x": {"value": 0.3, "unit": "m"},
                "y": {"value": 0, "unit": "m"},
            },
        ],
        "charges": [
            {"id": "q1", "charge": {"value": 2, "unit": "uC"}, "at": "A"},
            {"id": "q2", "charge": {"value": 3, "unit": "uC"}, "at": "B"},
        ],
        "queries": [
            {"type": "net_force", "target": "q2", "output": "magnitude", "unit": "N"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.domain == "electrostatics"
    assert result.answer == "0.6 N"


def test_unified_solver_rejects_unknown_domain():
    schema = {
        "domain": "unknown_domain",
        "queries": [],
    }

    result = solve_schema(schema)

    assert result.status == "solve_failed"
    assert "Unsupported domain" in result.error
