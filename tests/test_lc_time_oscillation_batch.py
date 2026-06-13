import math
import re

import pytest

from xai_physics.domains.equations.solver import solve_schema
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_lc_natural_period_from_l_and_c():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "L1", "type": "inductance", "role": "given", "value": "0.1", "unit": "H"},
            {"id": "C1", "type": "capacitance", "role": "given", "value": "1", "unit": "uF"},
            {"id": "T_query", "type": "period", "role": "query", "value": None, "unit": "ms"},
        ],
        "relations": [{"type": "formula", "name": "lc_natural_period", "objects": ["T_query", "L1", "C1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(1.98691765316, rel=1e-6)


def test_lc_angular_frequency_from_l_and_c():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "L1", "type": "inductance", "role": "given", "value": "4", "unit": "mH"},
            {"id": "C1", "type": "capacitance", "role": "given", "value": "250", "unit": "uF"},
            {"id": "omega_query", "type": "angular_frequency", "role": "query", "value": None, "unit": "rad/s"},
        ],
        "relations": [{"type": "formula", "name": "lc_resonance_angular_frequency", "objects": ["omega_query", "L1", "C1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(1000.0)


def test_lc_magnetic_energy_from_current_time():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "L1", "type": "inductance", "role": "given", "value": "0.4", "unit": "H"},
            {"id": "I0", "type": "current_amplitude", "role": "given", "value": "1.5", "unit": "A"},
            {"id": "omega1", "type": "angular_frequency", "role": "given", "value": "2000", "unit": "rad/s"},
            {"id": "t1", "type": "time", "role": "given", "value": "5e-4", "unit": "s"},
            {"id": "Wm_query", "type": "energy", "role": "query", "value": None, "unit": "J"},
        ],
        "relations": [{"type": "formula", "name": "lc_magnetic_energy_current_time", "objects": ["Wm_query", "L1", "I0", "omega1", "t1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    expected = 0.5 * 0.4 * (1.5 * math.cos(2000 * 5e-4)) ** 2
    assert _num(r.answer) == pytest.approx(expected, rel=1e-6)


def test_lc_max_magnetic_energy_from_current_amplitude():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "I0", "type": "current_amplitude", "role": "given", "value": "0.5", "unit": "A"},
            {"id": "L1", "type": "inductance", "role": "given", "value": "0.2", "unit": "H"},
            {"id": "Wmax_query", "type": "energy", "role": "query", "value": None, "unit": "J", "symbol": "Wm_max"},
        ],
        "relations": [{"type": "formula", "name": "lc_magnetic_energy_current_time", "objects": ["Wmax_query", "L1", "I0"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(0.025)


def test_lc_max_voltage_from_max_charge_and_capacitance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C1", "type": "capacitance", "role": "given", "value": "5", "unit": "uF"},
            {"id": "Qmax", "type": "charge_amplitude", "role": "given", "value": "2e-4", "unit": "C"},
            {"id": "Umax_query", "type": "voltage", "role": "query", "value": None, "unit": "V"},
        ],
        "relations": [{"type": "formula", "name": "lc_max_voltage_charge_capacitance", "objects": ["Umax_query", "Qmax", "C1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(40.0)


def test_lc_energy_complement_from_total_and_capacitor_voltage():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "W_total", "type": "total_energy", "role": "given", "value": "0.2", "unit": "J"},
            {"id": "C1", "type": "capacitance", "role": "given", "value": "20", "unit": "uF"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "100", "unit": "V"},
            {"id": "Wm_query", "type": "energy", "role": "query", "value": None, "unit": "J", "symbol": "Wm"},
        ],
        "relations": [{"type": "formula", "name": "lc_energy_complement", "objects": ["Wm_query", "W_total", "C1", "U1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(0.1)
