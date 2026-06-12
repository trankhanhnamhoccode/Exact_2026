from xai_physics.domains.electrostatics.engine import solve_schema


def test_engine_equilateral_triangle_schema_without_coordinates():
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

    result = solve_schema(schema)

    assert result.status == "solved"
    # Each force magnitude = k q q / r^2 = 9e9 * 1e-12 / 0.01 = 0.9 N
    # Angle between the two force vectors at C is 60 deg.
    # Resultant = sqrt(3) * 0.9 ≈ 1.5588457 N
    assert result.answer.startswith("1.55885")


def test_engine_collinear_schema_without_coordinates():
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
            {"id": "q1", "charge": {"value": 2, "unit": "uC"}, "at": "A"},
            {"id": "q2", "charge": {"value": 3, "unit": "uC"}, "at": "B"},
            {"id": "q3", "charge": {"value": 1, "unit": "uC"}, "at": "C"},
        ],
        "queries": [
            {"type": "net_force", "target": "q3", "output": "x_component", "unit": "N"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    # q1 on q3: 9e9*2e-12/(0.5^2)=0.072 N
    # q2 on q3: 9e9*3e-12/(0.3^2)=0.3 N
    # total along +x = 0.372 N
    assert result.answer == "0.372 N"
