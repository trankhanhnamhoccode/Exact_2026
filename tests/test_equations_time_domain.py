import re
import math

import pytest

from xai_physics.domains.equations.solver import solve_schema


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_harmonic_current_cos_time():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "I0", "type": "current_amplitude", "role": "given", "value": "2", "unit": "A"},
            {"id": "omega1", "type": "angular_frequency", "role": "given", "value": "100", "unit": "rad/s"},
            {"id": "t1", "type": "time", "role": "given", "value": "0.01", "unit": "s"},
            {"id": "phi1", "type": "phase", "role": "given", "value": "0", "unit": "rad"},
            {"id": "I_query", "type": "current", "role": "query", "value": None, "unit": "A"},
        ],
        "relations": [{"type": "formula", "name": "harmonic_current_cos_time", "objects": ["I_query", "I0", "omega1", "t1", "phi1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(2 * math.cos(1.0), rel=1e-9)


def test_harmonic_voltage_cos_time_with_degree_phase():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "U0", "type": "voltage_amplitude", "role": "given", "value": "100", "unit": "V"},
            {"id": "omega1", "type": "angular_frequency", "role": "given", "value": "0", "unit": "rad/s"},
            {"id": "t1", "type": "time", "role": "given", "value": "1", "unit": "s"},
            {"id": "phi1", "type": "phase", "role": "given", "value": "60", "unit": "deg"},
            {"id": "U_query", "type": "voltage", "role": "query", "value": None, "unit": "V"},
        ],
        "relations": [{"type": "formula", "name": "harmonic_voltage_cos_time", "objects": ["U_query", "U0", "omega1", "t1", "phi1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(50.0, abs=1e-9)


def test_harmonic_charge_cos_time():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Q0", "type": "charge_amplitude", "role": "given", "value": "20", "unit": "uC"},
            {"id": "omega1", "type": "angular_frequency", "role": "given", "value": "1000", "unit": "rad/s"},
            {"id": "t1", "type": "time", "role": "given", "value": "1", "unit": "ms"},
            {"id": "Q_query", "type": "charge", "role": "query", "value": None, "unit": "uC"},
        ],
        "relations": [{"type": "formula", "name": "harmonic_charge_cos_time", "objects": ["Q_query", "Q0", "omega1", "t1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(20 * math.cos(1.0), rel=1e-9)


def test_lc_electric_energy_time():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "W_total", "type": "energy", "role": "given", "value": "10", "unit": "mJ"},
            {"id": "omega1", "type": "angular_frequency", "role": "given", "value": "1000", "unit": "rad/s"},
            {"id": "t1", "type": "time", "role": "given", "value": "1", "unit": "ms"},
            {"id": "We_query", "type": "energy", "role": "query", "value": None, "unit": "mJ"},
        ],
        "relations": [{"type": "formula", "name": "lc_electric_energy_time", "objects": ["We_query", "W_total", "omega1", "t1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(10 * math.cos(1.0) ** 2, rel=1e-9)


def test_lc_magnetic_energy_time():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "W_total", "type": "energy", "role": "given", "value": "10", "unit": "mJ"},
            {"id": "omega1", "type": "angular_frequency", "role": "given", "value": "1000", "unit": "rad/s"},
            {"id": "t1", "type": "time", "role": "given", "value": "1", "unit": "ms"},
            {"id": "Wm_query", "type": "energy", "role": "query", "value": None, "unit": "mJ"},
        ],
        "relations": [{"type": "formula", "name": "lc_magnetic_energy_time", "objects": ["Wm_query", "W_total", "omega1", "t1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(10 * math.sin(1.0) ** 2, rel=1e-9)
