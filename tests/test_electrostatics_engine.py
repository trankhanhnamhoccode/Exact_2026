from xai_physics.domains.electrostatics.engine import solve_schema


def test_two_positive_charges_force_magnitude():
    schema = {
        "domain": "electrostatics",
        "points": [
            {"id": "A", "x": {"value": 0, "unit": "m"}, "y": {"value": 0, "unit": "m"}},
            {"id": "B", "x": {"value": 0.3, "unit": "m"}, "y": {"value": 0, "unit": "m"}},
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
    assert result.answer == "0.6 N"


def test_opposite_charges_force_x_component_negative():
    schema = {
        "domain": "electrostatics",
        "points": [
            {"id": "A", "x": {"value": 0, "unit": "m"}, "y": {"value": 0, "unit": "m"}},
            {"id": "B", "x": {"value": 0.3, "unit": "m"}, "y": {"value": 0, "unit": "m"}},
        ],
        "charges": [
            {"id": "q1", "charge": {"value": 2, "unit": "uC"}, "at": "A"},
            {"id": "q2", "charge": {"value": -3, "unit": "uC"}, "at": "B"},
        ],
        "queries": [
            {"type": "net_force", "target": "q2", "output": "x_component", "unit": "N"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "-0.6 N"


def test_three_charges_net_force_cancels():
    schema = {
        "domain": "electrostatics",
        "points": [
            {"id": "A", "x": {"value": -0.3, "unit": "m"}, "y": {"value": 0, "unit": "m"}},
            {"id": "B", "x": {"value": 0.3, "unit": "m"}, "y": {"value": 0, "unit": "m"}},
            {"id": "C", "x": {"value": 0, "unit": "m"}, "y": {"value": 0, "unit": "m"}},
        ],
        "charges": [
            {"id": "q1", "charge": {"value": 2, "unit": "uC"}, "at": "A"},
            {"id": "q2", "charge": {"value": 2, "unit": "uC"}, "at": "B"},
            {"id": "q3", "charge": {"value": 3, "unit": "uC"}, "at": "C"},
        ],
        "queries": [
            {"type": "net_force", "target": "q3", "output": "magnitude", "unit": "N"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "0 N"
