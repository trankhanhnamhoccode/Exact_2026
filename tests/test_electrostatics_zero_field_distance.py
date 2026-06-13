from __future__ import annotations

import json

from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.llm.client import SchemaLLMClient
from xai_physics.llm.schema_pipeline import solve_problem_with_llm
from xai_physics.schema_solver import solve_schema


class _BadElectricFieldLLM(SchemaLLMClient):
    def generate(self, prompt: str) -> str:
        return json.dumps(
            {
                "domain": "electrostatics",
                "charges": [
                    {"id": "q1", "charge": {"value": -9, "unit": "uC"}, "at": "A"},
                    {"id": "q2", "charge": {"value": 4, "unit": "uC"}, "at": "B"},
                ],
                "points": [{"id": "M", "coordinates": {"x": 0.2, "y": 0, "unit": "m"}}],
                "queries": [{"type": "electric_field", "target": "M", "unit": "V/m"}],
            }
        )


def _solve_formula_candidate(question: str) -> str:
    schemas = generate_equations_candidate_schemas(question)
    zero_schemas = [s for s in schemas if s["relations"][0]["name"] == "two_charge_zero_field_distance"]
    assert zero_schemas
    result = solve_schema(zero_schemas[0])
    assert result.status == "ok", result.error
    return str(result.answer)


def test_zero_field_same_sign_distance_from_b():
    answer = _solve_formula_candidate(
        "Given two point charges of the same sign and magnitudes q1 = 4q2, placed at A and B, "
        "separated by 12 cm. Find the point where the net electric field strength is zero, "
        "and calculate its distance from B."
    )

    assert answer == "4 cm"


def test_zero_field_opposite_sign_distance_from_a():
    answer = _solve_formula_candidate(
        "Two electric charges q1 = 9 x 10^-8 C and q2 = -16 x 10^-8 C are placed at points A "
        "and B, separated by 12 cm in the air. Find the point where the net electric field is zero, "
        "and calculate its distance from A."
    )

    assert answer == "36 cm"


def test_zero_field_coordinate_from_origin_for_opposite_sign_charges():
    answer = _solve_formula_candidate(
        "Given two point charges located along the Ox axis: charge q1 = -9 x 10^-6 C is placed "
        "at the origin O, and charge q2 = 4 x 10^-6 C is located 20 cm from the origin. What is "
        "the coordinate on the Ox axis where the electric field strength is zero?"
    )

    assert answer == "60 cm"


def test_pipeline_prefers_zero_field_distance_over_bad_electric_field_schema():
    question = (
        "Given two point charges located along the Ox axis: charge q1 = -9 x 10^-6 C is placed "
        "at the origin O, and charge q2 = 4 x 10^-6 C is located 20 cm from the origin. What is "
        "the coordinate on the Ox axis where the electric field strength is zero?"
    )

    output = solve_problem_with_llm(question, _BadElectricFieldLLM(), k=2)

    assert output.solve_result.status == "ok", output.solve_result.error
    assert output.solve_result.answer == "60 cm"
    assert output.schema is not None
    assert output.schema["relations"][0]["name"] == "two_charge_zero_field_distance"
