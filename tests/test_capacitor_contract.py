from xai_physics.domains.capacitor_state.contract import (
    SchemaValidationError,
    validate_schema,
)
from xai_physics.domains.capacitor_state.engine import solve_schema


def _valid_schema():
    return {
        "domain": "capacitor_state",
        "entities": [
            {
                "id": "C1",
                "type": "capacitor",
                "capacitance": {"value": 2, "unit": "uF"},
                "voltage": {"value": 100, "unit": "V"},
                "connected_to_source": False,
            },
            {
                "id": "C2",
                "type": "capacitor",
                "capacitance": {"value": 3, "unit": "uF"},
                "voltage": {"value": 0, "unit": "V"},
                "connected_to_source": False,
            },
        ],
        "events": [
            {
                "type": "ParallelRedistribution",
                "apply_to": ["C1", "C2"],
                "params": {"polarity": "same"},
            }
        ],
        "queries": [
            {
                "type": "voltage",
                "target": "system",
                "unit": "V",
            }
        ],
    }


def test_validate_schema_accepts_valid_parallel_schema():
    validate_schema(_valid_schema())


def test_validate_schema_rejects_unknown_entity_reference():
    schema = _valid_schema()
    schema["events"][0]["apply_to"] = ["C1", "C999"]

    try:
        validate_schema(schema)
    except SchemaValidationError as exc:
        assert "unknown capacitor" in str(exc)
    else:
        raise AssertionError("Expected SchemaValidationError")


def test_validate_schema_rejects_unsupported_unit():
    schema = _valid_schema()
    schema["entities"][0]["capacitance"]["unit"] = "banana"

    try:
        validate_schema(schema)
    except SchemaValidationError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("Expected SchemaValidationError")


def test_engine_returns_solve_failed_for_invalid_schema():
    schema = _valid_schema()
    schema["queries"][0]["target"] = "C999"

    result = solve_schema(schema)

    assert result.status == "solve_failed"
    assert "unknown capacitor" in result.error
