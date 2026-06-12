import re

import pytest

from xai_physics.domains.equations.solver import solve_schema


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_inductor_energy_from_inductance_current():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "L1", "type": "inductance", "role": "given", "value": "0.2", "unit": "H"},
            {"id": "I1", "type": "current", "role": "given", "value": "3", "unit": "A"},
            {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "mJ"},
        ],
        "relations": [{"type": "formula", "name": "inductor_energy", "objects": ["W_query", "L1", "I1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(900.0)


def test_inductor_current_from_energy_inductance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "W1", "type": "energy", "role": "given", "value": "1.8", "unit": "mJ"},
            {"id": "L1", "type": "inductance", "role": "given", "value": "0.3", "unit": "H"},
            {"id": "I_query", "type": "current", "role": "query", "value": None, "unit": "A"},
        ],
        "relations": [{"type": "formula", "name": "inductor_energy", "objects": ["I_query", "W1", "L1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(0.1095445115, rel=1e-9)


def test_lc_resonance_frequency_from_lc():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "L1", "type": "inductance", "role": "given", "value": "0.5", "unit": "H"},
            {"id": "C1", "type": "capacitance", "role": "given", "value": "5", "unit": "uF"},
            {"id": "f_query", "type": "frequency", "role": "query", "value": None, "unit": "Hz"},
        ],
        "relations": [{"type": "formula", "name": "lc_resonance_frequency", "objects": ["f_query", "L1", "C1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(100.6584242, rel=1e-7)


def test_lc_resonance_capacitance_from_frequency_inductance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "f1", "type": "frequency", "role": "given", "value": "400", "unit": "Hz"},
            {"id": "L1", "type": "inductance", "role": "given", "value": "0.01", "unit": "H"},
            {"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "uF"},
        ],
        "relations": [{"type": "formula", "name": "lc_resonance_frequency", "objects": ["C_query", "f1", "L1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(15.83143494, rel=1e-8)


def test_ohm_law_current_at_resonance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "120", "unit": "V"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "40", "unit": "Ω"},
            {"id": "I_query", "type": "current", "role": "query", "value": None, "unit": "A"},
        ],
        "relations": [{"type": "formula", "name": "ohm_law", "objects": ["I_query", "U1", "R1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(3.0)


def test_impedance_current_from_voltage_impedance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "80", "unit": "V"},
            {"id": "Z1", "type": "impedance", "role": "given", "value": "40", "unit": "Ω"},
            {"id": "I_query", "type": "current", "role": "query", "value": None, "unit": "A"},
        ],
        "relations": [{"type": "formula", "name": "impedance_voltage_current", "objects": ["I_query", "U1", "Z1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(2.0)


def test_power_voltage_resistance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "135", "unit": "V"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "90", "unit": "Ω"},
            {"id": "P_query", "type": "power", "role": "query", "value": None, "unit": "W"},
        ],
        "relations": [{"type": "formula", "name": "power_voltage_resistance", "objects": ["P_query", "U1", "R1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(202.5)


def test_power_current_resistance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "I1", "type": "current", "role": "given", "value": "3", "unit": "A"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "20", "unit": "Ω"},
            {"id": "P_query", "type": "power", "role": "query", "value": None, "unit": "W"},
        ],
        "relations": [{"type": "formula", "name": "power_current_resistance", "objects": ["P_query", "I1", "R1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(180.0)


def test_ac_impedance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "R1", "type": "resistance", "role": "given", "value": "30", "unit": "Ω"},
            {"id": "XL1", "type": "inductive_reactance", "role": "given", "value": "40", "unit": "Ω"},
            {"id": "XC1", "type": "capacitive_reactance", "role": "given", "value": "10", "unit": "Ω"},
            {"id": "Z_query", "type": "impedance", "role": "query", "value": None, "unit": "Ω"},
        ],
        "relations": [{"type": "formula", "name": "ac_impedance", "objects": ["Z_query", "R1", "XL1", "XC1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(42.42640687, rel=1e-8)


def test_power_factor():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "R1", "type": "resistance", "role": "given", "value": "30", "unit": "Ω"},
            {"id": "Z1", "type": "impedance", "role": "given", "value": "50", "unit": "Ω"},
            {"id": "cosphi_query", "type": "power_factor", "role": "query", "value": None, "unit": "-"},
        ],
        "relations": [{"type": "formula", "name": "power_factor", "objects": ["cosphi_query", "R1", "Z1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(0.6)


def test_quality_factor():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "L1", "type": "inductance", "role": "given", "value": "0.10", "unit": "H"},
            {"id": "C1", "type": "capacitance", "role": "given", "value": "50", "unit": "uF"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "40", "unit": "Ω"},
            {"id": "Qf_query", "type": "quality_factor", "role": "query", "value": None, "unit": "-"},
        ],
        "relations": [{"type": "formula", "name": "quality_factor", "objects": ["Qf_query", "L1", "C1", "R1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(1.118033989, rel=1e-9)


def test_frequency_scaling_for_resonance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "XL1", "type": "inductive_reactance", "role": "given", "value": "12", "unit": "Ω"},
            {"id": "XC1", "type": "capacitive_reactance", "role": "given", "value": "108", "unit": "Ω"},
            {"id": "factor_query", "type": "frequency_factor", "role": "query", "value": None, "unit": "times"},
        ],
        "relations": [{"type": "formula", "name": "frequency_scaling_for_resonance", "objects": ["factor_query", "XL1", "XC1"]}],
    }

    r = solve_schema(schema)

    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(3.0)
