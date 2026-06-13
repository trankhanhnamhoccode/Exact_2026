import math

from xai_physics.domains.electrostatics.engine import solve_schema
from xai_physics.domains.electrostatics.text_extractor import extract_electrostatics_schema_from_text
from xai_physics.domains.electrostatics.retrieval.pipeline import retrieve_electrostatics_context
from xai_physics.llm.schema_pipeline import solve_problem_with_llm


class FakeSchemaLLM:
    def __init__(self, output: str = "not json"):
        self.output = output
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.output


def _num(answer: str) -> float:
    return float(str(answer).split()[0])


def test_text_extractor_perpendicular_bisector_force_ld007():
    q = "Two charges, q1 = +2×10^-6 C and q2 = -2×10^-6 C, are placed at points A and B in a vacuum, 6 cm apart. A third charge, q3 = +2×10^-6 C, is placed on the perpendicular bisector of the line segment AB, 4 cm away from AB. What is the magnitude of the net electric force exerted by q1 and q2 on q3?"
    schema = extract_electrostatics_schema_from_text(q)
    assert schema is not None
    assert schema["geometry"][1]["type"] == "PerpendicularBisectorPoint"
    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 17.28) < 0.02


def test_text_extractor_midpoint_force_ld022():
    q = "Two electric charges, q1 = +2 μC and q2 = -2 μC, are placed 10 cm apart. A charge q3 = +1 μC is placed at the midpoint of the line segment connecting q1 and q2. Calculate the net force acting on q3."
    schema = extract_electrostatics_schema_from_text(q)
    assert schema is not None
    assert any(g["type"] == "Midpoint" for g in schema["geometry"])
    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 14.4) < 0.02


def test_text_extractor_equilateral_missing_geometry_ld012():
    q = "Three charges q1 = +1 μC, q2 = +1 μC, and q3 = -1 μC are placed at the vertices of an equilateral triangle with side a = 20 cm. Calculate the magnitude of the net force acting on q3."
    schema = extract_electrostatics_schema_from_text(q)
    assert schema is not None
    assert schema["geometry"][0]["type"] == "EquilateralTriangle"
    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 0.3897) < 0.01


def test_text_extractor_straight_line_equal_spacing_ld017():
    q = "Three charges: q1 = +3 μC, q2 = -2 μC, and q3 = +1 μC are placed 10 cm apart on a straight line. Calculate the force acting on q2."
    schema = extract_electrostatics_schema_from_text(q)
    assert schema is not None
    assert schema["geometry"][0]["type"] == "Collinear"
    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 3.6) < 0.01


def test_retrieval_prioritizes_perpendicular_bisector_force_example():
    q = "A charge q3 is on the perpendicular bisector of AB, 4 cm away from AB. q1 and q2 are at A and B, 6 cm apart. Find the force on q3."
    retrieval = retrieve_electrostatics_context(q, final_top_k=4)
    ids = {item.example.id for item in retrieval.selected_examples}
    assert "elec_ld007_perpendicular_bisector_force_point_c" in ids or "elec_perpendicular_bisector_force" in ids
    tags = set(retrieval.final_tags)
    assert "perpendicular_bisector" in tags
    assert "perpendicular_bisector_force" in tags


def test_pipeline_uses_deterministic_extractor_before_bad_llm():
    q = "Two charges, q1 = +2×10^-6 C and q2 = -2×10^-6 C, are placed at points A and B, 6 cm apart. A third charge, q3 = +2×10^-6 C, is placed on the perpendicular bisector of AB, 4 cm away from AB. Find the net force on q3."
    fake = FakeSchemaLLM("this would fail if called")
    out = solve_problem_with_llm(q, fake, k=2)
    assert fake.last_prompt is None
    assert out.schema is not None
    assert out.raw_llm_output == "__deterministic_electrostatics_text_extractor__"
    assert out.solve_result.status == "solved"
    assert abs(_num(out.solve_result.answer) - 17.28) < 0.02


def test_text_extractor_inverse_coulomb_equal_charges_ld014():
    q = "Two charges separated by 15 cm exert a force of 4.8 N. Given that q1 = q2 = q, find q."
    schema = extract_electrostatics_schema_from_text(q)
    assert schema is not None
    assert schema["queries"][0]["type"] == "coulomb_equal_charge"
    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 3.464) < 0.01


def test_text_extractor_inverse_resultant_angle_ld020():
    q = "A charge is acted upon by two forces, each of magnitude 10 N. Find the angle between the two forces if the resultant force is also 10 N."
    schema = extract_electrostatics_schema_from_text(q)
    assert schema is not None
    assert schema["queries"][0]["type"] == "resultant_angle"
    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 120.0) < 1e-6


def test_text_extractor_opposite_sides_target_centered_ld021():
    q = "A charge q = -1 μC is attracted by two +1 μC charges. These two positive charges are located on opposite sides of q, along the same straight line passing through q, at distances of 5 cm and 12 cm respectively from q. Calculate the magnitude of the net electric force acting on q."
    schema = extract_electrostatics_schema_from_text(q)
    assert schema is not None
    assert schema["geometry"][0]["type"] == "Collinear"
    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 2.975) < 0.02


def test_text_extractor_point_on_segment_one_distance_ld025():
    q = "Two charges, q1 = +2 μC and q2 = +2 μC, are placed at the ends of a 10 cm long line segment. A third charge, q3 = -1 μC, is positioned along the line connecting q1 and q2. Calculate the net force acting on q3 when it is 4 cm away from q1."
    schema = extract_electrostatics_schema_from_text(q)
    assert schema is not None
    assert any(g["type"] == "PointOnLine" for g in schema["geometry"])
    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 6.25) < 0.02


def test_text_extractor_pairwise_m_from_a_b_ld032():
    q = "In a vacuum, two charges q1 = 10^-7 C and q2 = -10^-7 C are placed at points A and B, separated by 8 cm. Determine the resultant force acting on a third charge q0 = 10^-7 C when q0 is placed at point M, 4 cm from A and 12 cm from B."
    schema = extract_electrostatics_schema_from_text(q)
    assert schema is not None
    assert schema["geometry"][0]["type"] == "PairwiseDistances"
    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 0.05) < 0.001


def test_text_extractor_equilateral_center_chain_charges_ld030():
    q = "Three charges, q1 = 8×10^-9 C, q2 = q3 = -8×10^-9 C, are placed at the three vertices of an equilateral triangle ABC with side length a = 6 cm, in air. Determine the net force acting on a charge q0 = 6×10^-9 C placed at the center O of the triangle."
    schema = extract_electrostatics_schema_from_text(q)
    assert schema is not None
    assert any(g["type"] == "Centroid" for g in schema["geometry"])
    result = solve_schema(schema)
    assert result.status == "solved"
    assert abs(_num(result.answer) - 7.2e-4) < 1e-6
