import re

import pytest

from xai_physics.domains.equations.solver import solve_schema


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_nl311_energy_scales_with_capacitance_at_constant_voltage():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C_ratio", "type": "ratio", "role": "given", "value": "2", "unit": "times", "symbol": "C2/C1"},
            {"id": "W_ratio_query", "type": "ratio", "role": "query", "value": None, "unit": "times", "symbol": "W2/W1"},
        ],
        "relations": [
            {
                "type": "formula",
                "name": "capacitor_energy_scaling_constant_voltage",
                "objects": ["C_ratio", "W_ratio_query"],
            }
        ],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(2.0)


def test_td365_energy_scales_with_voltage_squared_at_constant_capacitance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "V_ratio", "type": "ratio", "role": "given", "value": "3", "unit": "times", "symbol": "U2/U1"},
            {"id": "W_ratio_query", "type": "ratio", "role": "query", "value": None, "unit": "times", "symbol": "W2/W1"},
        ],
        "relations": [
            {
                "type": "formula",
                "name": "capacitor_energy_voltage_scaling_constant_capacitance",
                "objects": ["V_ratio", "W_ratio_query"],
            }
        ],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(9.0)


def test_td366_energy_scales_with_charge_squared_at_constant_capacitance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Q1", "type": "charge", "role": "given", "value": "90", "unit": "uC", "symbol": "Q1"},
            {"id": "Q2", "type": "charge", "role": "given", "value": "45", "unit": "uC", "symbol": "Q2"},
            {"id": "W_ratio_query", "type": "ratio", "role": "query", "value": None, "unit": "times", "symbol": "W2/W1"},
        ],
        "relations": [
            {
                "type": "formula",
                "name": "capacitor_energy_charge_scaling_constant_capacitance",
                "objects": ["Q1", "Q2", "W_ratio_query"],
            }
        ],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(0.25)


def test_series_capacitance_unknown_from_equivalent_and_known_capacitance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Ceq", "type": "capacitance", "role": "given", "value": "6", "unit": "uF", "symbol": "Ceq"},
            {"id": "C1", "type": "capacitance", "role": "given", "value": "10", "unit": "uF", "symbol": "C1"},
            {"id": "C2_query", "type": "capacitance", "role": "query", "value": None, "unit": "uF", "symbol": "C2"},
        ],
        "relations": [
            {
                "type": "formula",
                "name": "series_capacitance_unknown",
                "objects": ["Ceq", "C1", "C2_query"],
            }
        ],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(15.0)


def test_energy_scaling_percent_output():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C_ratio", "type": "ratio", "role": "given", "value": "0.5", "unit": "times", "symbol": "C2/C1"},
            {"id": "W_percent_query", "type": "ratio", "role": "query", "value": None, "unit": "%", "symbol": "W2/W1"},
        ],
        "relations": [
            {
                "type": "formula",
                "name": "capacitor_energy_scaling_constant_voltage",
                "objects": ["C_ratio", "W_percent_query"],
            }
        ],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(50.0)
