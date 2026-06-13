from __future__ import annotations

import json

import pytest

from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.llm.schema_pipeline import solve_problem_with_llm
from xai_physics.schema_solver import solve_schema


class FakeSchemaLLM:
    def __init__(self, output: str):
        self.output = output

    def generate(self, prompt: str) -> str:
        return self.output


def _bad_electro_schema() -> dict:
    return {
        "domain": "electrostatics",
        "points": [{"id": "A"}],
        "charges": [{"id": "q", "charge": {"value": 1, "unit": "nC"}, "at": "A"}],
        "queries": [{"type": "electric_field", "target": "M", "unit": "V/m"}],
    }


def test_point_charge_field_formula_candidate_from_electrostatics_text():
    problem = "A small sphere carrying an electric charge of 10^-9 C is placed in air. What is the electric field strength at a point 3cm away from the sphere?"

    candidates = generate_equations_candidate_schemas(problem)
    values = [float(str(solve_schema(schema).answer).split()[0]) for schema in candidates]

    assert any(value == pytest.approx(10000.0, rel=2e-3) for value in values)


def test_pipeline_uses_scalar_equations_fallback_for_point_charge_field():
    problem = "A small sphere carrying an electric charge of 10^-9 C is placed in air. What is the electric field strength at a point 3cm away from the sphere?"

    output = solve_problem_with_llm(problem, FakeSchemaLLM(json.dumps(_bad_electro_schema())), k=2)

    assert output.solve_result.status == "ok"
    assert output.schema is not None
    assert output.schema["domain"] == "equations"
    assert output.schema["relations"][0]["name"] == "point_charge_electric_field"
    assert float(str(output.solve_result.answer).split()[0]) == pytest.approx(10000.0, rel=2e-3)


def test_infinite_wire_field_formula_candidate():
    problem = "An infinitely long straight wire is uniformly charged with a linear charge density λ = -6 x 10^-9 C / m. Calculate the electric field strength at point M, which is a distance r = 20 cm from the wire."

    candidates = generate_equations_candidate_schemas(problem)
    values = [float(str(solve_schema(schema).answer).split()[0]) for schema in candidates]

    assert any(value == pytest.approx(540.0, rel=2e-3) for value in values)


def test_equilibrium_field_formula_candidate():
    problem = "A dust particle with a mass of 3.6 x 10^-15 kg, carrying an electric charge of 4.8 x 10^-18 C, is in equilibrium between two parallel plates. Calculate the electric field strength."

    candidates = generate_equations_candidate_schemas(problem)
    values = [float(str(solve_schema(schema).answer).split()[0]) for schema in candidates]

    assert any(value == pytest.approx(7500.0, rel=1e-9) for value in values)
