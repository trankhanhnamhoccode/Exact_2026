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


def _num(answer: object) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(answer))
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_right_triangle_known_sides_reconstructs_hypotenuse_geometry():
    problem = (
        "Three electric charges are placed at three fixed points, forming a right-angled triangle ABC "
        "(right-angled at A), where AB = 4 m and BC = 5 m. The charges are qA = 5.0 μC, "
        "qB = -5.0 μC, and qC = 4.0 μC, respectively. Find the net electric force acting on the charge at A."
    )

    candidates = generate_equations_candidate_schemas(problem)
    answers = [solve_schema(schema).answer for schema in candidates]

    assert any(_num(answer) == pytest.approx(24.45e-3, rel=2e-3) for answer in answers if answer)


def test_pipeline_prefers_geometry_derived_right_triangle_over_bad_isosceles_schema():
    problem = (
        "Three electric charges are placed at three fixed points, forming a right-angled triangle ABC "
        "(right-angled at A), where AB = 4 m and BC = 5 m. The charges are qA = 5.0 μC, "
        "qB = -5.0 μC, and qC = 4.0 μC, respectively. Find the net electric force acting on the charge at A."
    )
    bad_schema = {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "geometry": [{"type": "IsoscelesRightTriangle", "points": ["A", "B", "C"], "right_angle_at": "A", "leg": {"value": 4, "unit": "m"}}],
        "charges": [
            {"id": "q1", "charge": {"value": 5e-6, "unit": "C"}, "at": "A"},
            {"id": "q2", "charge": {"value": -5e-6, "unit": "C"}, "at": "B"},
            {"id": "q3", "charge": {"value": 4e-6, "unit": "C"}, "at": "C"},
        ],
        "queries": [{"type": "net_force", "target": "q1", "output": "magnitude", "unit": "N"}],
    }

    output = solve_problem_with_llm(problem, FakeSchemaLLM(json.dumps(bad_schema)), k=2)

    assert output.schema is not None
    assert output.schema["domain"] == "equations"
    assert output.schema["relations"][0]["name"] == "two_charge_geometry_field"
    assert output.solve_result.status == "ok"
    assert _num(output.solve_result.answer) == pytest.approx(24.45e-3, rel=2e-3)


def test_two_identical_charges_at_equilateral_vertices_keeps_both_sources():
    problem = (
        "Two identical charges q = +2 μC are placed at two vertices of an equilateral triangle with side length a = 0.1 m. "
        "A charge q′ = -1 μC is placed at the remaining vertex. Calculate the net electric force acting on q′."
    )

    candidates = generate_equations_candidate_schemas(problem)
    solved = [(schema, solve_schema(schema)) for schema in candidates]

    assert any(result.answer and _num(result.answer) == pytest.approx(3.1177, rel=2e-3) for _, result in solved)
    schema = next(schema for schema, result in solved if result.answer and _num(result.answer) == pytest.approx(3.1177, rel=2e-3))
    sources = [obj for obj in schema["objects"] if obj.get("type") == "charge" and obj.get("source")]
    assert len(sources) == 2
