import math
import re

import pytest

from xai_physics.domains.equations.retrieval.hard_rules import formula_rule_scores
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context
from xai_physics.domains.equations.solver import solve_schema
from xai_physics.domains.equations.text_extractor import extract_equations_schema_from_text


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", str(answer), flags=re.I)
    assert m, answer
    return float(m.group(0))


def test_resonance_resistance_from_impedance_schema_ch001():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Z1", "type": "impedance", "role": "given", "value": "40", "unit": "ohm"},
            {"id": "R_query", "type": "resistance", "role": "query", "value": None, "unit": "ohm"},
        ],
        "relations": [{"type": "formula", "name": "ac_impedance", "objects": ["R_query", "Z1"]}],
        "formula_candidates": ["ac_impedance", "resonance_check"],
    }
    result = solve_schema(schema)
    assert result.status == "ok", result.error
    assert _num(result.answer) == pytest.approx(40)


def test_resonance_impedance_from_resistance_schema_ch181():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "R1", "type": "resistance", "role": "given", "value": "30", "unit": "ohm"},
            {"id": "Z_query", "type": "impedance", "role": "query", "value": None, "unit": "ohm"},
        ],
        "relations": [{"type": "formula", "name": "ac_impedance", "objects": ["Z_query", "R1"]}],
        "formula_candidates": ["ac_impedance", "resonance_check"],
    }
    result = solve_schema(schema)
    assert result.status == "ok", result.error
    assert _num(result.answer) == pytest.approx(30)


def test_resonance_resistance_when_impedance_typed_as_resistance_ch007():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Z1", "type": "resistance", "role": "given", "value": "75", "unit": "ohm"},
            {"id": "R_query", "type": "resistance", "role": "query", "value": None, "unit": "ohm"},
        ],
        "relations": [{"type": "formula", "name": "ohm_law", "objects": ["Z1", "R_query"]}],
        "formula_candidates": ["resonance_check", "ohm_law"],
    }
    result = solve_schema(schema)
    assert result.status == "ok", result.error
    assert _num(result.answer) == pytest.approx(75)


def test_power_factor_at_resonance_even_when_schema_uses_phase_ch179():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "phi1", "type": "phase", "role": "given", "value": "0", "unit": ""},
            {"id": "cos_phi_query", "type": "phase", "role": "query", "value": None, "unit": ""},
        ],
        "relations": [{"type": "formula", "name": "resonance_check", "objects": ["phi1", "cos_phi_query"]}],
        "formula_candidates": ["resonance_check"],
    }
    result = solve_schema(schema)
    assert result.status == "ok", result.error
    assert result.answer.strip() == "1"


def test_frequency_scaling_from_reactance_pair_even_when_formula_bad_ch203():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "XL1", "type": "inductive_reactance", "role": "given", "value": "36", "unit": "ohm"},
            {"id": "XC1", "type": "capacitive_reactance", "role": "given", "value": "144", "unit": "ohm"},
            {"id": "omega_query", "type": "angular_frequency", "role": "query", "value": None, "unit": ""},
        ],
        "relations": [{"type": "formula", "name": "resonance_check", "objects": ["XL1", "XC1", "omega_query"]}],
        "formula_candidates": ["resonance_check", "ac_inductive_reactance", "ac_capacitive_reactance"],
    }
    result = solve_schema(schema)
    assert result.status == "ok", result.error
    assert _num(result.answer) == pytest.approx(2.0)


def test_text_extractor_inverse_capacitance_ch062():
    schema = extract_equations_schema_from_text("What capacitance must the capacitor have for an L-C circuit with L = 0.2 H to resonate at 100 Hz?")
    assert schema is not None
    result = solve_schema(schema)
    assert result.status == "ok", result.error
    assert _num(result.answer) == pytest.approx(12.665, rel=2e-3)


def test_text_extractor_inverse_inductance_ch066():
    schema = extract_equations_schema_from_text("A capacitor with C=40 μF. To resonate at f=100 Hz, what inductor L is needed?")
    assert schema is not None
    result = solve_schema(schema)
    assert result.status == "ok", result.error
    assert _num(result.answer) == pytest.approx(0.06333, rel=2e-3)


def test_text_extractor_direct_f0_ch344():
    schema = extract_equations_schema_from_text("A circuit has L = 0.08 H and C = 100 µF. Calculate f0.")
    assert schema is not None
    result = solve_schema(schema)
    assert result.status == "ok", result.error
    assert _num(result.answer) == pytest.approx(56.27, rel=2e-3)


def test_hard_rules_prioritize_resonance_identity_and_inverse():
    scores = formula_rule_scores("At resonance, the total impedance of the circuit is measured as Z = 120 Ω. What is R?")
    assert scores["rlc_resonance_impedance_resistance"] >= 9.0
    scores = formula_rule_scores("What capacitance C is needed for a circuit to resonate at f = 400 Hz with L = 0.01 H?")
    assert scores["lc_resonance_frequency"] >= 7.5


def test_retrieval_returns_resonance_identity_doc():
    ctx = retrieve_equations_context("At resonance Z = 40 ohm. Find R.", formula_top_k=5, example_top_k=2)
    ids = [item.formula.id for item in ctx.selected_formulas]
    assert "rlc_resonance_impedance_resistance" in ids
