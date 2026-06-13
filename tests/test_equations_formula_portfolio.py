from __future__ import annotations

import json

from xai_physics.domains.equations.solver import solve_schema
from xai_physics.llm.schema_pipeline import solve_problem_with_llm


class FakeSchemaLLM:
    def __init__(self, output: str):
        self.output = output
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.output


def test_formula_portfolio_tries_schema_formula_candidates_after_bad_relation():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C1", "type": "capacitance", "role": "given", "value": "4", "unit": "uF"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "6", "unit": "V"},
            {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "uJ"},
        ],
        "relations": [
            {"type": "formula", "name": "not_a_real_formula", "objects": ["C1", "U1", "W_query"]}
        ],
        "formula_candidates": ["capacitor_energy_voltage"],
    }

    result = solve_schema(schema)

    assert result.status == "ok"
    assert result.answer == "72 uJ"
    assert any(step.title == "Formula portfolio selected" for step in result.trace)


def test_formula_portfolio_infers_compatible_formula_from_objects_without_candidates():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Q1", "type": "charge", "role": "given", "value": "40", "unit": "uC"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "8", "unit": "V"},
            {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "uJ"},
        ],
        "relations": [
            {"type": "formula", "name": "capacitor_charge_voltage", "objects": ["Q1", "U1", "W_query"]}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "ok"
    assert result.answer == "160 uJ"


def test_llm_pipeline_attaches_retrieval_formula_candidates_for_equations():
    problem = "A capacitor with a capacitance of 4 μF is charged to a voltage of 6 V. Calculate the electric field energy of the capacitor."
    bad_schema = {
        "domain": "equations",
        "objects": [
            {"id": "C1", "type": "capacitance", "role": "given", "value": "4", "unit": "uF"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "6", "unit": "V"},
            {"id": "u_query", "type": "energy_density", "role": "query", "value": None, "unit": "J/m3"},
        ],
        "relations": [
            {"type": "formula", "name": "capacitor_energy_density", "objects": ["U1", "C1", "u_query"]}
        ],
    }
    fake = FakeSchemaLLM(json.dumps(bad_schema))

    output = solve_problem_with_llm(problem, fake, k=2)

    assert output.schema is not None
    assert "formula_candidates" in output.schema
    assert "capacitor_energy_voltage" in output.schema["formula_candidates"]
    assert output.solve_result.status == "ok"
    assert output.solve_result.answer == "72 uJ"
