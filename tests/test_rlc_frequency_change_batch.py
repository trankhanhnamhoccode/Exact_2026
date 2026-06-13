import re

import pytest

from xai_physics.domains.equations.solver import solve_schema
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_frequency_factor_for_resonance_from_reactances():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "XL1", "type": "inductive_reactance", "role": "given", "value": "36", "unit": "ohm"},
            {"id": "XC1", "type": "capacitive_reactance", "role": "given", "value": "144", "unit": "ohm"},
            {"id": "k_query", "type": "frequency_factor", "role": "query", "value": None, "unit": ""},
        ],
        "relations": [{"type": "formula", "name": "frequency_scaling_for_resonance", "objects": ["k_query", "XL1", "XC1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(2.0)


def test_resistor_voltage_after_frequency_double_without_resistance_when_resonant():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "XL1", "type": "inductive_reactance", "role": "given", "value": "25", "unit": "ohm"},
            {"id": "XC1", "type": "capacitive_reactance", "role": "given", "value": "100", "unit": "ohm"},
            {"id": "k1", "type": "frequency_factor", "role": "given", "value": "2", "unit": ""},
            {"id": "U1", "type": "voltage", "role": "given", "value": "120", "unit": "V"},
            {"id": "UR_query", "type": "resistor_voltage", "role": "query", "value": None, "unit": "V", "symbol": "UR"},
        ],
        "relations": [{"type": "formula", "name": "rlc_frequency_scaled_response", "objects": ["UR_query", "XL1", "XC1", "k1", "U1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(120.0)


def test_current_after_frequency_double_with_resistance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "XL1", "type": "inductive_reactance", "role": "given", "value": "25", "unit": "ohm"},
            {"id": "XC1", "type": "capacitive_reactance", "role": "given", "value": "100", "unit": "ohm"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "40", "unit": "ohm"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "120", "unit": "V"},
            {"id": "k1", "type": "frequency_factor", "role": "given", "value": "2", "unit": ""},
            {"id": "I_query", "type": "current", "role": "query", "value": None, "unit": "A"},
        ],
        "relations": [{"type": "formula", "name": "rlc_frequency_scaled_response", "objects": ["I_query", "XL1", "XC1", "R1", "U1", "k1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(3.0)


def test_power_after_frequency_triple_with_resistance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "XL1", "type": "inductive_reactance", "role": "given", "value": "15", "unit": "ohm"},
            {"id": "XC1", "type": "capacitive_reactance", "role": "given", "value": "135", "unit": "ohm"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "25", "unit": "ohm"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "100", "unit": "V"},
            {"id": "k1", "type": "frequency_factor", "role": "given", "value": "3", "unit": ""},
            {"id": "P_query", "type": "power", "role": "query", "value": None, "unit": "W"},
        ],
        "relations": [{"type": "formula", "name": "rlc_frequency_scaled_response", "objects": ["P_query", "XL1", "XC1", "R1", "U1", "k1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(400.0)


def test_inductor_voltage_at_resonance_from_l_c_r_u():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "120", "unit": "V"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "40", "unit": "ohm"},
            {"id": "L1", "type": "inductance", "role": "given", "value": "0.10", "unit": "H"},
            {"id": "C1", "type": "capacitance", "role": "given", "value": "50", "unit": "uF"},
            {"id": "UL_query", "type": "inductor_voltage", "role": "query", "value": None, "unit": "V", "symbol": "UL"},
        ],
        "relations": [{"type": "formula", "name": "rlc_component_voltage_at_resonance", "objects": ["UL_query", "U1", "R1", "L1", "C1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(134.16407865, rel=1e-6)


def test_frequency_change_retrieval_prioritizes_scaled_response():
    q = "A circuit consists of XL = 25 Ω, XC = 100 Ω, R = 40 Ω, U = 120 V. If the frequency is increased by 2 times, what is the effective current in the circuit?"
    ctx = retrieve_equations_context(q, formula_top_k=5, example_top_k=5)
    formula_ids = [item.formula.id for item in ctx.selected_formulas]
    example_ids = [item.example.id for item in ctx.selected_examples]
    assert formula_ids[0] == "rlc_frequency_scaled_response"
    assert "eq_ex_rlc_current_after_frequency_double_ch265" in example_ids


def test_resonance_component_voltage_retrieval_prioritizes_ul_formula():
    q = "Given U = 120 V, R = 40 Ω, L = 0.10 H, C = 50 µF at resonance. Calculate UL, the voltage across the inductor."
    ctx = retrieve_equations_context(q, formula_top_k=5, example_top_k=5)
    formula_ids = [item.formula.id for item in ctx.selected_formulas]
    example_ids = [item.example.id for item in ctx.selected_examples]
    assert formula_ids[0] == "rlc_component_voltage_at_resonance"
    assert "eq_ex_rlc_inductor_voltage_at_resonance_ch365" in example_ids
