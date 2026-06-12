import re

import pytest

from xai_physics.domains.equations.solver import solve_schema


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_capacitor_plate_force_by_charge_area():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Q1", "type": "charge", "role": "given", "value": "2", "unit": "uC"},
            {"id": "A1", "type": "area", "role": "given", "value": "100", "unit": "cm2"},
            {"id": "F_query", "type": "force", "role": "query", "value": None, "unit": "N"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_plate_force_by_charge_area", "objects": ["F_query", "Q1", "A1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(22.5881813367, rel=1e-8)


def test_self_inductance_from_emf():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "emf1", "type": "voltage", "role": "given", "value": "12", "unit": "V"},
            {"id": "dt1", "type": "time", "role": "given", "value": "5", "unit": "ms"},
            {"id": "dI1", "type": "current_change", "role": "given", "value": "2", "unit": "A"},
            {"id": "L_query", "type": "inductance", "role": "query", "value": None, "unit": "mH"},
        ],
        "relations": [{"type": "formula", "name": "self_inductance_from_emf", "objects": ["L_query", "emf1", "dt1", "dI1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(30.0)


def test_equilibrium_mass_with_angle_default_g_10():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "q1", "type": "charge", "role": "given", "value": "2", "unit": "uC"},
            {"id": "E1", "type": "electric_field", "role": "given", "value": "5000", "unit": "V/m"},
            {"id": "theta1", "type": "angle", "role": "given", "value": "45", "unit": "deg"},
            {"id": "m_query", "type": "mass", "role": "query", "value": None, "unit": "g"},
        ],
        "relations": [{"type": "formula", "name": "equilibrium_mass_with_angle", "objects": ["m_query", "q1", "E1", "theta1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(1.0)


def test_lc_current_amplitude_from_charge_amplitude():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "omega1", "type": "angular_frequency", "role": "given", "value": "1000", "unit": "rad/s"},
            {"id": "Q0", "type": "charge_amplitude", "role": "given", "value": "20", "unit": "uC"},
            {"id": "I0_query", "type": "current_amplitude", "role": "query", "value": None, "unit": "mA"},
        ],
        "relations": [{"type": "formula", "name": "lc_current_amplitude_from_charge_amplitude", "objects": ["I0_query", "omega1", "Q0"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(20.0)
