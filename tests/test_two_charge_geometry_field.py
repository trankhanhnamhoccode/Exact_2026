from __future__ import annotations

from xai_physics.eval.replay_llm_dataset import compare_answer
from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.schema_solver import solve_schema


def _solve_two_charge_geometry(question: str) -> str:
    schemas = generate_equations_candidate_schemas(question)
    matches = [schema for schema in schemas if schema["relations"][0]["name"] == "two_charge_geometry_field"]
    assert matches, [schema["relations"][0]["name"] for schema in schemas]
    result = solve_schema(matches[0])
    assert result.status == "ok", result.error
    return str(result.answer)


def test_two_charge_triangle_field_from_equal_distances():
    answer = _solve_two_charge_geometry(
        "Two point charges, q1 = q2 = 16 x 10^-8 C, are placed at points A and B, "
        "which are 10 cm apart in the air. Determine the electric field strength "
        "caused by these two point charges at point C, given that AC = BC = 8 cm."
    )

    assert compare_answer(answer, "351000", expected_unit="V/m") is True


def test_two_charge_geometry_returns_force_on_test_charge():
    answer = _solve_two_charge_geometry(
        "At two points A and B, 10 cm apart in the air, two point charges are placed: "
        "q1 = 6 x 10^-6 C and q2 = -6 x 10^-6 C. Determine the electric field "
        "strength caused by these two charges at point C, given that AC = BC = 12 cm. "
        "Calculate the electric force acting on a charge q3 = -3 x 10^-8 C placed at C."
    )

    assert compare_answer(answer, "0.094", expected_unit="N") is True


def test_two_charge_perpendicular_bisector_uses_components():
    answer = _solve_two_charge_geometry(
        "Two charges, q1 = 5.3 × 10^-6 C and q2 = -3.0 × 10^-6 C, are separated by "
        "9.2 cm. Point M is on the perpendicular bisector of the line segment connecting "
        "the charges, 4.6 cm from its midpoint. Calculate the net electric field strength at M."
    )

    assert compare_answer(answer, "1.3*10^7", expected_unit="V/m") is True


def test_two_charge_collinear_outside_left_of_q1():
    answer = _solve_two_charge_geometry(
        "Two charges, q1 = -2.4 × 10^-6 C and q2 = -2.1 × 10^-6 C, are separated by "
        "7.5 cm. Point M lies on the line connecting the two charges but outside the "
        "segment between them, and is located 11.9 cm to the left of charge q1. "
        "Calculate the net electric field at M."
    )

    assert compare_answer(answer, "2.027*10^6", expected_unit="V/m") is True
