from __future__ import annotations

from xai_physics.domains.electrostatics.text_extractor import extract_electrostatics_schema_from_text
from xai_physics.schema_solver import solve_schema


def test_equal_charge_chain_q1_q2_q3_keeps_all_charges_and_force_intent():
    problem = (
        "Three electric charges q1 = q2 = q3 = +8 × 10^-6 C are placed "
        "at the three vertices of an equilateral triangle with side length 15 cm in the air. "
        "Calculate the net electric force acting on q3."
    )

    schema = extract_electrostatics_schema_from_text(problem)

    assert schema is not None
    assert {ch["id"] for ch in schema["charges"]} == {"q1", "q2", "q3"}
    assert schema["queries"][0]["type"] == "net_force"
    assert schema["queries"][0]["target"] == "q3"
    result = solve_schema(schema)
    assert result.status == "solved"
    assert result.answer == "44.3405 N"
