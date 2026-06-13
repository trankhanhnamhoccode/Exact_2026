from __future__ import annotations

from xai_physics.eval.replay_llm_dataset import compare_answer
from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.schema_solver import solve_schema


def _solved(problem: str):
    for schema in generate_equations_candidate_schemas(problem):
        result = solve_schema(schema)
        if result.status in {"ok", "solved"} and result.answer:
            yield schema, str(result.answer)


def test_symbolic_force_resultant_uses_formula_not_literal_answer():
    problem = (
        "Two charges of the same magnitude q are placed at two adjacent vertices of an isosceles right triangle with side length a = 10 cm. "
        "Find the magnitude of the total electric force acting on a test charge q0 placed at the remaining vertex, given F0 = kqq0/a^2."
    )
    solved = list(_solved(problem))
    assert any(schema["relations"][0]["name"] == "symbolic_equal_perpendicular_resultant" for schema, _ in solved)
    assert any(compare_answer(ans, r"\sqrt{2} × F₀", expected_unit="N") is True for _, ans in solved)


def test_conceptual_direction_is_derived_from_charge_signs_and_between_geometry():
    problem = (
        "A test charge is placed at a point whose distances to the two charges q1 = +2 μC and q2 = -3 μC are 3 cm and 4 cm, respectively. "
        "The two charges are fixed and separated by 7 cm. What is the direction of the net electric force acting on the test charge?"
    )
    solved = list(_solved(problem))
    assert any(schema["relations"][0]["name"] == "direction_between_collinear_charges" for schema, _ in solved)
    assert any(compare_answer(ans, "Hướng về phía q₂", expected_unit="-") is True for _, ans in solved)


def test_symbolic_field_ratio_from_force_and_charge_ratios():
    problem = (
        "Two test charges q1 and q2 (q1 = 4q2) are placed at points A and B respectively in an electric field. "
        "The force acting on q1 is F1, and the force acting on q2 is F2 (with F1 = 3F2). "
        "Let E1 and E2 be the electric field strengths at A and B respectively. What is the relationship between E1 and E2?"
    )
    solved = list(_solved(problem))
    assert any(schema["relations"][0]["name"] == "symbolic_field_ratio_from_force_charge_ratios" for schema, _ in solved)
    assert any(compare_answer(ans, "E1 = (3/4)E2", expected_unit="-") is True for _, ans in solved)


def test_symbolic_right_isosceles_altitude_field_expression():
    problem = (
        "At the three vertices of a right isosceles triangle ABC, with AB = AC = a, three positive charges qA = qB = q and qC = 2q are placed in a vacuum. "
        "What is the expression for the electric field intensity at H, which is the foot of the altitude dropped from the right-angle vertex A to the hypotenuse BC?"
    )
    solved = list(_solved(problem))
    assert any(schema["relations"][0]["name"] == "symbolic_right_isosceles_altitude_field" for schema, _ in solved)
    assert any(compare_answer(ans, "2 × sqrt(2) × k × q / a^2", expected_unit="-") is True for _, ans in solved)


def test_symbolic_square_missing_charge_expression():
    problem = (
        "Given a square ABCD with side length 'a'. Charges q1 = q3 = q are placed at A and C. "
        "What charge must be placed at B so that the electric field at D is zero?"
    )
    solved = list(_solved(problem))
    assert any(schema["relations"][0]["name"] == "symbolic_square_field_zero_missing_charge" for schema, _ in solved)
    assert any(compare_answer(ans, r"-2\sqrt{2} x q", expected_unit="-") is True for _, ans in solved)
