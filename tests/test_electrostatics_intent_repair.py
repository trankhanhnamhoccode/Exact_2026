from __future__ import annotations

from xai_physics.domains.electrostatics.engine import solve_schema
from xai_physics.hybrid.electrostatics_repair import repair_electrostatics_schema_with_question


def test_repair_force_question_from_electric_field_query_to_net_force():
    problem = (
        "Three charges q1 = +2 μC, q2 = +2 μC, and q3 = -2 μC are placed "
        "at the three vertices of an equilateral triangle with a side length of 10 cm. "
        "Calculate the magnitude of the net electric force acting on q3."
    )
    bad_schema = {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "geometry": [
            {
                "type": "EquilateralTriangle",
                "points": ["A", "B", "C"],
                "side": {"value": 10, "unit": "cm"},
                "orientation": "above",
            }
        ],
        "charges": [
            {"id": "q1", "charge": {"value": 2, "unit": "uC"}, "at": "A"},
            {"id": "q2", "charge": {"value": 2, "unit": "uC"}, "at": "B"},
            {"id": "q3", "charge": {"value": -2, "unit": "uC"}, "at": "C"},
        ],
        "queries": [{"type": "electric_field", "target": "C", "output": "magnitude", "unit": "V/m"}],
    }

    repaired = repair_electrostatics_schema_with_question(problem, bad_schema)

    assert repaired["queries"][0] == {"type": "net_force", "target": "q3", "output": "magnitude", "unit": "N"}
    result = solve_schema(repaired)
    assert result.status == "solved"
    assert result.answer == "6.23538 N"


def test_repair_does_not_change_real_electric_field_question():
    problem = "Calculate the resultant electric field strength at M."
    schema = {
        "domain": "electrostatics",
        "charges": [{"id": "q1", "charge": {"value": 1, "unit": "uC"}, "at": "A"}],
        "queries": [{"type": "electric_field", "target": "M", "unit": "V/m"}],
    }

    assert repair_electrostatics_schema_with_question(problem, schema) is schema
