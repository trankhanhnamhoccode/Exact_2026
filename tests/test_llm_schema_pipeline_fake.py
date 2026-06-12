from __future__ import annotations

import json

from xai_physics.llm.json_extractor import extract_json_object
from xai_physics.llm.schema_pipeline import solve_problem_with_llm


class FakeSchemaLLM:
    def __init__(self, output: str):
        self.output = output
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.output


def _capacitor_energy_schema() -> dict:
    return {
        "domain": "equations",
        "objects": [
            {"id": "C1", "type": "capacitance", "role": "given", "value": "100", "unit": "uF"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "30", "unit": "V"},
            {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "J"},
        ],
        "relations": [
            {"type": "formula", "name": "capacitor_energy_voltage", "objects": ["W_query", "C1", "U1"]}
        ],
        "constraints": [],
    }


def test_extract_json_object_from_plain_json():
    schema = _capacitor_energy_schema()

    parsed = extract_json_object(json.dumps(schema))

    assert parsed["domain"] == "equations"
    assert parsed["relations"][0]["name"] == "capacitor_energy_voltage"


def test_extract_json_object_from_fenced_json():
    schema = _capacitor_energy_schema()
    raw = "```json\n" + json.dumps(schema) + "\n```"

    parsed = extract_json_object(raw)

    assert parsed["domain"] == "equations"
    assert parsed["objects"][0]["type"] == "capacitance"


def test_extract_json_object_from_text_wrapped_response():
    schema = _capacitor_energy_schema()
    raw = "Sure, here is the JSON:\n" + json.dumps(schema) + "\nDone."

    parsed = extract_json_object(raw)

    assert parsed["domain"] == "equations"


def test_fake_llm_pipeline_solves_equations_schema():
    problem = "Calculate the energy stored in a capacitor with capacitance 100 uF and voltage 30 V."
    fake = FakeSchemaLLM(json.dumps(_capacitor_energy_schema()))

    output = solve_problem_with_llm(problem, fake, k=2)

    assert fake.last_prompt is not None
    assert "Domain:\nequations" in fake.last_prompt
    assert "Relevant formula docs:" in fake.last_prompt
    assert "capacitor_energy_voltage" in fake.last_prompt

    assert output.schema is not None
    assert output.solve_result.status == "ok"
    assert output.solve_result.domain == "equations"
    assert output.solve_result.answer == "0.045 J"


def test_fake_llm_pipeline_accepts_fenced_json():
    problem = "Calculate the energy stored in a capacitor with capacitance 100 uF and voltage 30 V."
    raw = "```json\n" + json.dumps(_capacitor_energy_schema()) + "\n```"
    fake = FakeSchemaLLM(raw)

    output = solve_problem_with_llm(problem, fake, k=2)

    assert output.schema is not None
    assert output.solve_result.status == "ok"
    assert output.solve_result.answer == "0.045 J"


def test_fake_llm_pipeline_returns_solve_failed_on_invalid_json():
    problem = "Calculate the energy stored in a capacitor with capacitance 100 uF and voltage 30 V."
    fake = FakeSchemaLLM("not json at all")

    output = solve_problem_with_llm(problem, fake, k=2)

    assert output.schema is None
    assert output.solve_result.status == "solve_failed"
    assert output.solve_result.domain == "equations"
    assert "Could not" in output.solve_result.error


def test_fake_llm_pipeline_can_route_capacitor_state_schema():
    problem = (
        "A capacitor is charged to 300 V, then disconnected and a dielectric constant 2 is inserted. "
        "Find the final voltage."
    )
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                    "capacitance": {"value": 500, "unit": "pF"},
                "voltage": {"value": 300, "unit": "V"},
                "connected_to_source": True,
            }
        ],
        "events": [
            {"type": "DisconnectFromSource", "apply_to": ["C1"], "params": {}},
            {"type": "InsertDielectric", "apply_to": ["C1"], "params": {"dielectric_constant": 2}},
        ],
        "queries": [{"type": "voltage", "target": "C1", "unit": "V"}],
    }
    fake = FakeSchemaLLM(json.dumps(schema))

    output = solve_problem_with_llm(problem, fake, k=2)

    assert fake.last_prompt is not None
    assert "Domain:\ncapacitor_state" in fake.last_prompt
    assert output.solve_result.status == "solved"
    assert output.solve_result.answer == "150 V"
