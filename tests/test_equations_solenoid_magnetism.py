import re

import pytest

from xai_physics.domains.equations.solver import solve_schema


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_solenoid_turn_density():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "N1", "type": "turn_count", "role": "given", "value": "1000", "unit": "turns"},
            {"id": "l1", "type": "length", "role": "given", "value": "0.5", "unit": "m"},
            {"id": "n_query", "type": "turn_density", "role": "query", "value": None, "unit": "turns/m"},
        ],
        "relations": [{"type": "formula", "name": "solenoid_turn_density", "objects": ["n_query", "N1", "l1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(2000.0)


def test_solenoid_magnetic_field_from_turn_count_length_current():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "l1", "type": "length", "role": "given", "value": "0.5", "unit": "m"},
            {"id": "N1", "type": "turn_count", "role": "given", "value": "1000", "unit": "turns"},
            {"id": "I1", "type": "current", "role": "given", "value": "2", "unit": "A"},
            {"id": "B_query", "type": "magnetic_field", "role": "query", "value": None, "unit": "T"},
        ],
        "relations": [{"type": "formula", "name": "solenoid_magnetic_field", "objects": ["B_query", "N1", "I1", "l1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(0.005026548246, rel=1e-9)


def test_solenoid_magnetic_field_from_turn_density_current():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "n1", "type": "turn_density", "role": "given", "value": "2000", "unit": "turns/m"},
            {"id": "I1", "type": "current", "role": "given", "value": "2", "unit": "A"},
            {"id": "B_query", "type": "magnetic_field", "role": "query", "value": None, "unit": "mT"},
        ],
        "relations": [{"type": "formula", "name": "solenoid_magnetic_field_from_n", "objects": ["B_query", "n1", "I1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(5.026548246, rel=1e-9)


def test_solenoid_inductance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "N1", "type": "turn_count", "role": "given", "value": "1200", "unit": "turns"},
            {"id": "A1", "type": "area", "role": "given", "value": "50", "unit": "cm2"},
            {"id": "l1", "type": "length", "role": "given", "value": "0.6", "unit": "m"},
            {"id": "L_query", "type": "inductance", "role": "query", "value": None, "unit": "mH"},
        ],
        "relations": [{"type": "formula", "name": "solenoid_inductance", "objects": ["L_query", "N1", "A1", "l1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(15.07964474, rel=1e-8)


def test_magnetic_flux_total():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "N1", "type": "turn_count", "role": "given", "value": "1000", "unit": "turns"},
            {"id": "B1", "type": "magnetic_field", "role": "given", "value": "0.01", "unit": "T"},
            {"id": "A1", "type": "area", "role": "given", "value": "20", "unit": "cm2"},
            {"id": "Phi_query", "type": "magnetic_flux", "role": "query", "value": None, "unit": "Wb"},
        ],
        "relations": [{"type": "formula", "name": "magnetic_flux_total", "objects": ["Phi_query", "N1", "B1", "A1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(0.02)


def test_magnetic_energy_density():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "B1", "type": "magnetic_field", "role": "given", "value": "0.1", "unit": "T"},
            {"id": "u_query", "type": "energy_density", "role": "query", "value": None, "unit": "J/m3"},
        ],
        "relations": [{"type": "formula", "name": "magnetic_energy_density", "objects": ["u_query", "B1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(3978.873577, rel=1e-9)
