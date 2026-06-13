from xai_physics.domains.equations.solver import solve_schema


def test_electric_field_energy_wording_mislabeled_as_density_is_repaired():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C1", "type": "capacitance", "role": "given", "value": "4", "unit": "uF"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "6", "unit": "V"},
            {"id": "u_query", "type": "energy_density", "role": "query", "value": None, "unit": "J/m3"},
        ],
        "relations": [
            {"type": "formula", "name": "capacitor_energy_density", "objects": ["U1", "C1", "u_query"]}
        ],
        "constraints": [],
    }

    result = solve_schema(schema)

    assert result.status == "ok"
    assert result.answer is not None
    assert str(result.answer).startswith("72")
    assert "uJ" in str(result.answer) or "μJ" in str(result.answer) or "µJ" in str(result.answer)
