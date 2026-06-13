from __future__ import annotations

import pytest

from xai_physics.hybrid.formula_filler import Quantity, fill_formula_specs
from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.schema_solver import solve_schema


def test_formula_spec_filler_emits_capacitor_charge_voltage_schema():
    schemas = fill_formula_specs(
        {
            "capacitance": (Quantity("20", "pF"),),
            "voltage": (Quantity("60", "V"),),
        },
        "charge",
    )

    assert [schema["relations"][0]["name"] for schema in schemas] == ["capacitor_charge_voltage"]
    result = solve_schema(schemas[0])
    assert result.status == "ok"
    assert float(str(result.answer).split()[0]) == pytest.approx(1.2)


def test_formula_spec_filler_emits_electric_force_field_schema():
    schemas = fill_formula_specs(
        {
            "force": (Quantity("3", "mN"),),
            "charge": (Quantity("1e-7", "C"),),
        },
        "electric_field",
    )

    names = [schema["relations"][0]["name"] for schema in schemas]
    assert "electric_force_field" in names
    result = solve_schema(schemas[names.index("electric_force_field")])
    assert result.status == "ok"
    assert float(str(result.answer).split()[0]) == pytest.approx(3.0e4)


def test_equations_candidate_generator_keeps_formula_spec_candidates_as_extension():
    problem = "A charge q = 10^-7 C experiences a force F = 3 mN. Calculate the electric field strength."

    schemas = generate_equations_candidate_schemas(problem)
    names = [schema["relations"][0]["name"] for schema in schemas]

    assert "electric_force_field" in names
