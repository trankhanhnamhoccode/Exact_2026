from xai_physics.domains.electrostatics.coordinate_builder import build_coordinates
from xai_physics.domains.electrostatics.engine import solve_schema
from xai_physics.llm.prompt_builder import build_schema_prompt


def _num(answer: str) -> float:
    return float(str(answer).split()[0])


def test_pairwise_distances_ld003_schema_solves():
    schema = {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "geometry": [
            {
                "type": "PairwiseDistances",
                "points": ["A", "B", "C"],
                "distances": [
                    {"between": ["A", "B"], "value": 20, "unit": "cm"},
                    {"between": ["A", "C"], "value": 12, "unit": "cm"},
                    {"between": ["B", "C"], "value": 16, "unit": "cm"},
                ],
                "orientation": "above",
            }
        ],
        "charges": [
            {"id": "q1", "charge": {"value": -3e-6, "unit": "C"}, "at": "A"},
            {"id": "q2", "charge": {"value": 8e-6, "unit": "C"}, "at": "B"},
            {"id": "q3", "charge": {"value": 2e-6, "unit": "C"}, "at": "C"},
        ],
        "queries": [{"type": "net_force", "target": "q3", "output": "magnitude", "unit": "N"}],
    }

    coords = build_coordinates(schema)
    assert abs(coords["C"].x - 0.072) < 1e-12
    assert abs(coords["C"].y - 0.096) < 1e-12

    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 6.76) < 0.01


def test_midpoint_force_cancels_for_equal_charges():
    schema = {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": "B"}, {"id": "M"}],
        "geometry": [
            {"type": "PairwiseDistances", "points": ["A", "B"], "distances": [{"between": ["A", "B"], "value": 4, "unit": "cm"}]},
            {"type": "Midpoint", "point": "M", "between": ["A", "B"]},
        ],
        "charges": [
            {"id": "q1", "charge": {"value": 1, "unit": "uC"}, "at": "A"},
            {"id": "q2", "charge": {"value": 1, "unit": "uC"}, "at": "B"},
            {"id": "q0", "charge": {"value": 2, "unit": "uC"}, "at": "M"},
        ],
        "queries": [{"type": "net_force", "target": "q0", "output": "magnitude", "unit": "N"}],
    }

    result = solve_schema(schema)
    assert result.status == "solved"
    assert result.answer == "0 N"


def test_perpendicular_bisector_force_schema_solves():
    schema = {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": "B"}, {"id": "M"}],
        "geometry": [
            {"type": "PairwiseDistances", "points": ["A", "B"], "distances": [{"between": ["A", "B"], "value": 6, "unit": "cm"}]},
            {"type": "PerpendicularBisectorPoint", "point": "M", "between": ["A", "B"], "distance_from_segment": {"value": 3, "unit": "cm"}},
        ],
        "charges": [
            {"id": "q1", "charge": {"value": 2e-6, "unit": "C"}, "at": "A"},
            {"id": "q2", "charge": {"value": 2e-6, "unit": "C"}, "at": "B"},
            {"id": "q", "charge": {"value": 1e-6, "unit": "C"}, "at": "M"},
        ],
        "queries": [{"type": "net_force", "target": "q", "output": "magnitude", "unit": "N"}],
    }

    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 14.1421) < 0.01


def test_electric_field_at_charge_position_excludes_self_charge():
    schema = {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "geometry": [{"type": "EquilateralTriangle", "points": ["A", "B", "C"], "side": {"value": 10, "unit": "cm"}}],
        "charges": [
            {"id": "q1", "charge": {"value": 1, "unit": "uC"}, "at": "A"},
            {"id": "q2", "charge": {"value": 1, "unit": "uC"}, "at": "B"},
            {"id": "q3", "charge": {"value": 1, "unit": "uC"}, "at": "C"},
        ],
        "queries": [{"type": "electric_field", "target": "q3", "output": "magnitude", "unit": "V/m"}],
    }

    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 1558845.7) < 10


def test_resultant_vector_schema_solves_direct_force_problem():
    schema = {
        "domain": "electrostatics",
        "vectors": [
            {"id": "F1", "magnitude": {"value": 5, "unit": "N"}, "angle_deg": 0},
            {"id": "F2", "magnitude": {"value": 12, "unit": "N"}, "angle_deg": 60},
        ],
        "queries": [{"type": "resultant_vector", "target": "vectors", "output": "magnitude", "unit": "N"}],
    }

    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 15.1327) < 0.001


def test_electrostatics_prompt_retrieves_examples():
    problem = "Points A and B are separated by 20 cm. q1 is at A, q2 is at B, and q3 is at C with AC = 12 cm and BC = 16 cm. Find the net force on q3."
    built = build_schema_prompt(problem, k=2)

    assert built.domain_decision.domain == "electrostatics"
    assert built.examples
    assert "PairwiseDistances" in built.prompt
    assert "elec_pairwise_triangle_ld003" in {ex["id"] for ex in built.examples}
