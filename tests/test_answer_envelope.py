from __future__ import annotations

import math

from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.llm.schema_pipeline import solve_problem_with_llm
from xai_physics.schema_solver import solve_schema


class FakeSchemaLLM:
    def __init__(self, output: str = "{}"):
        self.output = output

    def generate(self, prompt: str) -> str:
        return self.output


def _solve_first_formula(problem: str, formula: str):
    for schema in generate_equations_candidate_schemas(problem):
        if schema.get("relations", [{}])[0].get("name") != formula:
            continue
        result = solve_schema(schema)
        if result.status in {"ok", "solved"}:
            return result
    raise AssertionError(f"No solved candidate for {formula}")


def test_numeric_answer_meta_for_two_charge_field_candidate():
    question = (
        "Two electric charges q1 = +3 × 10^-6 C and q2 = +4 × 10^-6 C are placed at "
        "points A and B, which are 5 cm apart. Calculate the resultant electric field "
        "strength at point M, which is 3 cm from A and 4 cm from B."
    )
    out = solve_problem_with_llm(question, FakeSchemaLLM(), k=2)
    meta = out.solve_result.answer_meta

    assert out.solve_result.status in {"ok", "solved"}
    assert out.solve_result.answer == "37500000 V/m"
    assert meta is not None
    assert meta.answer_type == "numeric"
    assert meta.quantity_type == "electric_field"
    assert meta.unit == "V/m"
    assert math.isclose(meta.numeric_value_si or 0.0, 37_500_000, rel_tol=1e-12)


def test_symbolic_answer_meta_for_perpendicular_resultant():
    question = (
        "Two charges of the same magnitude q are placed at two adjacent vertices of an "
        "isosceles right triangle with side length a = 10 cm. Find the magnitude of the "
        "total electric force acting on a test charge q0 placed at the remaining vertex, "
        "given F0 = kqq0/a^2."
    )
    out = solve_problem_with_llm(question, FakeSchemaLLM(), k=2)
    meta = out.solve_result.answer_meta

    assert out.solve_result.status in {"ok", "solved"}
    assert out.solve_result.answer == "sqrt(2) × F0"
    assert meta is not None
    assert meta.answer_type == "symbolic"
    assert meta.unit == "N"
    assert meta.variables == {"F0": "k*q*q0/a^2"}
    assert meta.symbolic_canonical is not None


def test_direction_answer_meta_for_symbolic_direction():
    problem = (
        "A test charge is placed at a point whose distances to the two charges q1 = +2 μC and q2 = -3 μC are 3 cm and 4 cm, respectively. "
        "The two charges are fixed and separated by 7 cm. What is the direction of the net electric force acting on the test charge?"
    )
    result = _solve_first_formula(problem, "direction_between_collinear_charges")
    meta = result.answer_meta

    assert result.answer == "Hướng về phía q₂"
    assert meta is not None
    assert meta.answer_type == "direction"
    assert meta.direction.get("target") == "q2"
