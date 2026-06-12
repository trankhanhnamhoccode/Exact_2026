from xai_physics.domains.electrostatics.coordinate_builder import build_coordinates


def test_build_equilateral_triangle_coordinates():
    schema = {
        "domain": "electrostatics",
        "points": [
            {"id": "A"},
            {"id": "B"},
            {"id": "C"},
        ],
        "geometry": [
            {
                "type": "EquilateralTriangle",
                "points": ["A", "B", "C"],
                "side": {"value": 10, "unit": "cm"},
                "orientation": "above",
            }
        ],
        "charges": [
            {"id": "q1", "charge": {"value": 1, "unit": "uC"}, "at": "A"},
            {"id": "q2", "charge": {"value": 1, "unit": "uC"}, "at": "B"},
            {"id": "q3", "charge": {"value": 1, "unit": "uC"}, "at": "C"},
        ],
        "queries": [
            {"type": "net_force", "target": "q3", "output": "magnitude", "unit": "N"}
        ],
    }

    coords = build_coordinates(schema)

    assert abs(coords["A"].x - 0.0) < 1e-12
    assert abs(coords["A"].y - 0.0) < 1e-12
    assert abs(coords["B"].x - 0.1) < 1e-12
    assert abs(coords["B"].y - 0.0) < 1e-12
    assert abs(coords["C"].x - 0.05) < 1e-12


def test_build_collinear_coordinates():
    schema = {
        "domain": "electrostatics",
        "points": [
            {"id": "A"},
            {"id": "B"},
            {"id": "C"},
        ],
        "geometry": [
            {
                "type": "Collinear",
                "points": ["A", "B", "C"],
                "order": ["A", "B", "C"],
                "distances": [
                    {"between": ["A", "B"], "value": 20, "unit": "cm"},
                    {"between": ["B", "C"], "value": 30, "unit": "cm"},
                ],
            }
        ],
        "charges": [
            {"id": "q1", "charge": {"value": 1, "unit": "uC"}, "at": "A"},
            {"id": "q2", "charge": {"value": 1, "unit": "uC"}, "at": "B"},
            {"id": "q3", "charge": {"value": 1, "unit": "uC"}, "at": "C"},
        ],
        "queries": [
            {"type": "net_force", "target": "q3", "output": "magnitude", "unit": "N"}
        ],
    }

    coords = build_coordinates(schema)

    assert abs(coords["A"].x - 0.0) < 1e-12
    assert abs(coords["B"].x - 0.2) < 1e-12
    assert abs(coords["C"].x - 0.5) < 1e-12
