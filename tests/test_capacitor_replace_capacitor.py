from xai_physics.domains.capacitor_state.contract import validate_schema
from xai_physics.domains.capacitor_state.engine import solve_schema
from xai_physics.domains.capacitor_state.retrieval.pipeline import retrieve_capacitor_context
from xai_physics.llm.domain_classifier import classify_domain


def test_replace_capacitor_same_voltage_energy_reduction_state():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 8, "unit": "uF"},
                "voltage": {"value": 10, "unit": "V"},
                "connected_to_source": True,
            }
        ],
        "events": [
            {
                "type": "ReplaceCapacitor",
                "apply_to": ["C1"],
                "params": {
                    "new_capacitance": {"value": 4, "unit": "uF"},
                    "hold": "voltage",
                },
            }
        ],
        "queries": [
            {"type": "energy_reduction", "target": "C1", "unit": "uJ"}
        ],
    }

    validate_schema(schema)
    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "200 uJ"


def test_replace_capacitor_bad_dielectric_schema_is_repaired():
    schema = {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 8, "unit": "uF"},
                "voltage": {"value": 10, "unit": "V"},
            },
            {
                "id": "C2",
                "type": "capacitor",
                "capacitance": {"value": 4, "unit": "uF"},
                "voltage": {"value": 10, "unit": "V"},
            },
        ],
        "events": [
            {"type": "ReplaceDielectric", "apply_to": ["C1"], "params": {}},
            {
                "type": "ParallelRedistribution",
                "apply_to": ["C1", "C2"],
                "params": {"polarity": "same"},
            },
        ],
        "queries": [
            {"type": "energy_ratio", "target": "C1", "unit": "times"}
        ],
    }

    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.answer == "200 uJ"


def test_replace_capacitor_retrieval_prefers_replace_capacitor_not_dielectric():
    q = (
        "A capacitor has a voltage of 10 V and a capacitance of 8 μF. "
        "If it's replaced by another capacitor with a capacitance of 4 μF, "
        "while maintaining the same voltage, what is the reduction in energy?"
    )

    decision = classify_domain(q)
    assert decision.domain == "capacitor_state"
    assert "replace_capacitor" in decision.tags
    assert "replace_dielectric" not in decision.tags

    ctx = retrieve_capacitor_context(q, vector_top_k=8, final_top_k=4)
    assert "replace_capacitor" in ctx.final_tags
    assert "energy_reduction_query" in ctx.final_tags
    assert any(
        item.example.id == "cap_td373_replace_cap_same_voltage_energy_reduction"
        for item in ctx.selected_examples
    )
