import re

import pytest

from xai_physics.domains.equations.solver import solve_schema


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_point_charge_electric_field():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "q1", "type": "charge", "role": "given", "value": "2", "unit": "uC"},
            {"id": "r1", "type": "distance", "role": "given", "value": "30", "unit": "cm"},
            {"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"},
        ],
        "relations": [{"type": "formula", "name": "point_charge_electric_field", "objects": ["E_query", "q1", "r1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(199723.37316, rel=1e-8)


def test_electric_force_from_charge_field():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "q1", "type": "charge", "role": "given", "value": "5", "unit": "uC"},
            {"id": "E1", "type": "electric_field", "role": "given", "value": "2000", "unit": "V/m"},
            {"id": "F_query", "type": "force", "role": "query", "value": None, "unit": "mN"},
        ],
        "relations": [{"type": "formula", "name": "electric_force_field", "objects": ["F_query", "q1", "E1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(10.0)


def test_electric_field_from_force_charge():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "F1", "type": "force", "role": "given", "value": "10", "unit": "mN"},
            {"id": "q1", "type": "charge", "role": "given", "value": "5", "unit": "uC"},
            {"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"},
        ],
        "relations": [{"type": "formula", "name": "electric_force_field", "objects": ["E_query", "F1", "q1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(2000.0)


def test_equilibrium_electric_field_default_g_10():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "m1", "type": "mass", "role": "given", "value": "2", "unit": "g"},
            {"id": "q1", "type": "charge", "role": "given", "value": "4", "unit": "uC"},
            {"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"},
        ],
        "relations": [{"type": "formula", "name": "equilibrium_electric_field", "objects": ["E_query", "m1", "q1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(5000.0)


def test_infinite_wire_electric_field():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "lambda1", "type": "line_charge_density", "role": "given", "value": "3", "unit": "uC/m"},
            {"id": "r1", "type": "distance", "role": "given", "value": "20", "unit": "cm"},
            {"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"},
        ],
        "relations": [{"type": "formula", "name": "infinite_wire_electric_field", "objects": ["E_query", "lambda1", "r1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(269626.553769, rel=1e-9)


def test_percentage_relative_error():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "err1", "type": "absolute_error", "role": "given", "value": "0.2", "unit": ""},
            {"id": "measured1", "type": "measured_value", "role": "given", "value": "10", "unit": ""},
            {"id": "percent_query", "type": "percent_error", "role": "query", "value": None, "unit": "%"},
        ],
        "relations": [{"type": "formula", "name": "percentage_relative_error", "objects": ["percent_query", "err1", "measured1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(2.0)


def test_absolute_error_from_actual():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "actual1", "type": "actual_value", "role": "given", "value": "9.8", "unit": "m"},
            {"id": "measured1", "type": "measured_value", "role": "given", "value": "10.0", "unit": "m"},
            {"id": "err_query", "type": "absolute_error", "role": "query", "value": None, "unit": "m"},
        ],
        "relations": [{"type": "formula", "name": "absolute_error_from_actual", "objects": ["err_query", "actual1", "measured1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(0.2)
