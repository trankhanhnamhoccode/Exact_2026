from __future__ import annotations

import json
import re

import pytest

from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.llm.schema_pipeline import solve_problem_with_llm
from xai_physics.schema_solver import solve_schema


class FakeSchemaLLM:
    def __init__(self, output: str):
        self.output = output
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.output


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(answer))
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_formula_candidate_extracts_symbolic_c_u_energy_pattern():
    problem = "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."

    candidates = generate_equations_candidate_schemas(problem)
    answers = [solve_schema(schema).answer for schema in candidates]

    assert any(_num(ans) == pytest.approx(0.045) for ans in answers if ans)


def test_hybrid_pipeline_recovers_parallel_plate_charge_from_bad_llm_schema():
    problem = (
        "A parallel-plate capacitor has circular plates with a radius of 10 cm. "
        "The distance between the plates and the potential difference across them are "
        "1 cm and 108 V, respectively. The space between the plates is air. "
        "What is the charge on the capacitor?"
    )
    bad_llm_schema = {
        "domain": "equations",
        "objects": [
            {"id": "C1", "type": "capacitance", "role": "constant", "value": None, "unit": "uF"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "108", "unit": "V"},
            {"id": "Q_query", "type": "charge", "role": "query", "value": None, "unit": "uC"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_charge_voltage", "objects": ["Q_query", "C1", "U1"]}],
        "constraints": [],
    }

    output = solve_problem_with_llm(problem, FakeSchemaLLM(json.dumps(bad_llm_schema)), k=2)

    assert output.solve_result.status == "ok", output.solve_result.error
    assert output.schema is not None
    assert output.schema["relations"][0]["name"] == "parallel_plate_charge_from_voltage"
    assert _num(output.solve_result.answer) == pytest.approx(3.00415515113, rel=1e-8)
    assert any(step.title == "Hybrid schema candidate selected" for step in output.solve_result.trace)


def test_hybrid_pipeline_prefers_dielectric_constant_query_over_wrong_capacitance_query():
    problem = (
        "A parallel-plate capacitor with a capacitance of 7.0 nF is filled with a dielectric. "
        "The area of each plate is 15 cm² and the distance between the plates is 10⁻⁵ m. "
        "What is the dielectric constant of the dielectric?"
    )
    bad_llm_schema = {
        "domain": "equations",
        "objects": [
            {"id": "A1", "type": "area", "role": "given", "value": "15.0", "unit": "cm2"},
            {"id": "d1", "type": "distance", "role": "given", "value": "1e-05", "unit": "m"},
            {"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "nF"},
        ],
        "relations": [{"type": "formula", "name": "parallel_plate_capacitance", "objects": ["A1", "d1", "C_query"]}],
        "constraints": [],
    }

    output = solve_problem_with_llm(problem, FakeSchemaLLM(json.dumps(bad_llm_schema)), k=2)

    assert output.solve_result.status == "ok", output.solve_result.error
    assert output.schema is not None
    query_types = {obj["type"] for obj in output.schema["objects"] if obj.get("role") == "query"}
    assert "relative_permittivity" in query_types
    assert _num(output.solve_result.answer) == pytest.approx(5.27057564524, rel=1e-8)
