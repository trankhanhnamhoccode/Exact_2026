from __future__ import annotations

from xai_physics.eval.replay_llm_dataset import compare_answer
from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.schema_solver import solve_schema


def _solve_vector_resultant(question: str) -> str:
    schemas = generate_equations_candidate_schemas(question)
    matches = [schema for schema in schemas if schema["relations"][0]["name"] == "two_field_vector_resultant"]
    assert matches, [schema["relations"][0]["name"] for schema in schemas]
    result = solve_schema(matches[0])
    assert result.status == "ok", result.error
    return str(result.answer)


def test_two_field_vector_resultant_with_sixty_degree_angle():
    answer = _solve_vector_resultant(
        "Two electric charges, q1 = 5.3 × 10^-6 C and q2 = 3.0 × 10^-6 C, "
        "are placed such that each is 5.0 cm from point M. The angle formed by "
        "the two charges at point M is 60°. Calculate the resultant electric field "
        "strength at M."
    )

    assert compare_answer(answer, "2.62 × 10^7", expected_unit="V/m") is True


def test_two_field_vector_resultant_with_perpendicular_fields():
    answer = _solve_vector_resultant(
        "Two charges, q1 = 4.98 × 10^-6 C and q2 = 4.74 × 10^-6 C, are located "
        "such that each is 4.56 cm away from point M. The electric fields they "
        "produce at M are perpendicular to each other. Calculate the magnitude of "
        "the resultant electric field at M."
    )

    assert compare_answer(answer, "2.98*10^7", expected_unit="V/m") is True


def test_two_field_vector_resultant_parses_are_located_from_m_wording():
    answer = _solve_vector_resultant(
        "Two electric charges, q1 = 4.10 × 10^-6 C and q2 = 3.68 × 10^-6 C, "
        "are located 3.47 cm from point M. The electric fields they produce at M "
        "form an angle of 60° with each other. Calculate the magnitude of the total "
        "electric field at M."
    )

    assert compare_answer(answer, "50.39 × 10^6", expected_unit="V/m") is True
