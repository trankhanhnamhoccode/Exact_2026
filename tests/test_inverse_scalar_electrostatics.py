from __future__ import annotations

from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.schema_solver import solve_schema


def _solve_by_formula(question: str, formula: str) -> str:
    schemas = generate_equations_candidate_schemas(question)
    matches = [schema for schema in schemas if schema["relations"][0]["name"] == formula]
    assert matches, [schema["relations"][0]["name"] for schema in schemas]
    result = solve_schema(matches[0])
    assert result.status == "ok", result.error
    return str(result.answer)


def test_point_charge_inverse_with_dielectric_and_direction():
    answer = _solve_by_formula(
        "A charge q is placed in a dielectric medium. At point M, 40 cm away from q, "
        "the electric field has a magnitude of 9 x 10^5 V / m and is directed towards "
        "the charge q. Given that the dielectric constant of the medium is 2.5. "
        "Determine the sign and magnitude of q.",
        "point_charge_electric_field",
    )

    assert answer.startswith("-4.005")
    assert answer.endswith(" C")


def test_coulomb_force_unknown_source_charge():
    answer = _solve_by_formula(
        "A charge q = 10^-7 C is placed in the electric field of a point charge Q, "
        "experiencing a force F = 3 mN. Calculate the magnitude of charge Q, given "
        "that the two charges are separated by a distance of 30 cm in vacuum.",
        "coulomb_force_two_charges",
    )

    assert answer.startswith("3.004")
    assert answer.endswith("e-07 C")


def test_zero_field_plus_charge_sum_solves_q1_and_q2():
    q1 = _solve_by_formula(
        "Two point charges q1 and q2 are placed at A and B, with AB = 2 cm. Given "
        "that q1 + q2 = 7 x 10^-8 C, and at point M, which is 6 cm from q1 and "
        "8 cm from q2, the electric field strength is E = 0. Find q1.",
        "two_charge_zero_field_unknown_charges",
    )
    q2 = _solve_by_formula(
        "Two point charges q1 and q2 are placed at A and B, with AB = 2 cm. Given "
        "that q1 + q2 = 7 x 10^-8 C. At point M, which is 6 cm from q1 and "
        "8 cm from q2, the net electric field strength is E = 0. Find q2.",
        "two_charge_zero_field_unknown_charges",
    )

    assert q1 == "-9e-08 C"
    assert q2 == "1.6e-07 C"


def test_charged_pendulum_deflection_angle():
    answer = _solve_by_formula(
        "A small sphere of mass m = 25 g, carrying an electric charge q = 2.5 x 10⁻⁷ C, "
        "is suspended by an inextensible string. It is placed in a uniform electric field "
        "with a horizontal electric field strength of magnitude E = 10⁶ V/m. Take g = 10 m/s². "
        "What is the angle of deflection of the string?",
        "electric_pendulum_deflection_angle",
    )

    assert answer == "0.785398163397 rad"


def test_midpoint_field_from_two_field_values():
    answer = _solve_by_formula(
        "The electric field strength produced by a point charge at point A is 36 V / m, "
        "and at point B is 9 V / m. Points A and B lie on the same electric field line. "
        "What is the electric field strength at point C, the midpoint of AB?",
        "midpoint_field_from_two_field_values",
    )

    assert answer == "16 V/m"
