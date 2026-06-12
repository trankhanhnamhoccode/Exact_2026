import re

import pytest

from xai_physics.domains.equations.solver import solve_schema


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_td401_capacitor_energy_from_capacitance_voltage():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C1", "type": "capacitance", "role": "given", "value": "100", "unit": "uF"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "30", "unit": "V"},
            {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "J"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_energy_voltage", "objects": ["W_query", "C1", "U1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(0.045)


def test_td402_capacitance_from_charge_voltage():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Q1", "type": "charge", "role": "given", "value": "3", "unit": "mC"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "30", "unit": "V"},
            {"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "uF"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_charge_voltage", "objects": ["Q1", "C_query", "U1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(100.0)


def test_td363_voltage_from_charge_capacitance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C1", "type": "capacitance", "role": "given", "value": "1", "unit": "uF"},
            {"id": "Q1", "type": "charge", "role": "given", "value": "10", "unit": "uC"},
            {"id": "V_query", "type": "voltage", "role": "query", "value": None, "unit": "V"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_charge_voltage", "objects": ["Q1", "C1", "V_query"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(10.0)


def test_td364_voltage_from_energy_capacitance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C1", "type": "capacitance", "role": "given", "value": "4", "unit": "uF"},
            {"id": "W1", "type": "energy", "role": "given", "value": "1", "unit": "mJ"},
            {"id": "V_query", "type": "voltage", "role": "query", "value": None, "unit": "V"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_energy_voltage", "objects": ["W1", "C1", "V_query"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(22.360679775, rel=1e-9)


def test_td369_capacitance_from_energy_voltage():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "W1", "type": "energy", "role": "given", "value": "0.36", "unit": "J"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "120", "unit": "V"},
            {"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "uF"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_energy_voltage", "objects": ["W1", "C_query", "U1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(50.0)


def test_td168_parallel_plate_capacitance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "A1", "type": "area", "role": "given", "value": "20.3", "unit": "cm2"},
            {"id": "d1", "type": "distance", "role": "given", "value": "0.61", "unit": "mm"},
            {"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "pF"},
        ],
        "relations": [{"type": "formula", "name": "parallel_plate_capacitance", "objects": ["C_query", "A1", "d1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(29.46557585, rel=1e-6)


def test_td093_parallel_plate_dielectric_constant():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C1", "type": "capacitance", "role": "given", "value": "5.2", "unit": "nF"},
            {"id": "A1", "type": "area", "role": "given", "value": "12.3", "unit": "cm2"},
            {"id": "d1", "type": "distance", "role": "given", "value": "1.2e-5", "unit": "m"},
            {"id": "eps_query", "type": "relative_permittivity", "role": "query", "value": None, "unit": "-"},
        ],
        "relations": [{"type": "formula", "name": "parallel_plate_capacitance", "objects": ["eps_query", "C1", "A1", "d1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(5.729685022, rel=1e-6)


def test_td012_parallel_plate_charge_from_breakdown_field_and_radius():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "r1", "type": "radius", "role": "given", "value": "60", "unit": "cm"},
            {"id": "Emax", "type": "electric_field", "role": "given", "value": "3e5", "unit": "V/m"},
            {"id": "Q_query", "type": "charge", "role": "query", "value": None, "unit": "uC"},
        ],
        "relations": [{"type": "formula", "name": "parallel_plate_charge_from_field", "objects": ["Q_query", "r1", "Emax"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(3.00415515113, rel=1e-6)


def test_td397_parallel_plate_charge_from_voltage():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "A1", "type": "area", "role": "given", "value": "400", "unit": "cm2"},
            {"id": "d1", "type": "distance", "role": "given", "value": "2", "unit": "mm"},
            {"id": "eps1", "type": "relative_permittivity", "role": "given", "value": "1.5", "unit": "-"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "100", "unit": "V"},
            {"id": "Q_query", "type": "charge", "role": "query", "value": None, "unit": "nC"},
        ],
        "relations": [{"type": "formula", "name": "parallel_plate_charge_from_voltage", "objects": ["Q_query", "A1", "d1", "eps1", "U1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(26.56256345, rel=1e-6)


def test_td392_electric_field_energy_density_from_voltage_distance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "eps1", "type": "relative_permittivity", "role": "given", "value": "2", "unit": "-"},
            {"id": "d1", "type": "distance", "role": "given", "value": "0.5", "unit": "mm"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "100", "unit": "V"},
            {"id": "u_query", "type": "energy_density", "role": "query", "value": None, "unit": "J/m3"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_energy_density", "objects": ["u_query", "eps1", "U1", "d1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(0.35416751268, rel=1e-9)
