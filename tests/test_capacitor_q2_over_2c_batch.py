import re

import pytest

from xai_physics.domains.equations.solver import solve_schema
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_capacitor_energy_from_charge_and_capacitance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Q1", "type": "charge", "role": "given", "value": "20", "unit": "uC"},
            {"id": "C1", "type": "capacitance", "role": "given", "value": "5", "unit": "uF"},
            {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "uJ"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_energy_charge_capacitance", "objects": ["W_query", "Q1", "C1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(40.0)


def test_capacitor_charge_from_energy_and_capacitance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C1", "type": "capacitance", "role": "given", "value": "5", "unit": "uF"},
            {"id": "W1", "type": "energy", "role": "given", "value": "40", "unit": "uJ"},
            {"id": "Q_query", "type": "charge", "role": "query", "value": None, "unit": "uC"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_energy_charge_capacitance", "objects": ["Q_query", "W1", "C1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(20.0)


def test_capacitor_capacitance_from_energy_and_charge():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Q1", "type": "charge", "role": "given", "value": "20", "unit": "uC"},
            {"id": "W1", "type": "energy", "role": "given", "value": "40", "unit": "uJ"},
            {"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "uF"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_energy_charge_capacitance", "objects": ["C_query", "Q1", "W1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(5.0)


def test_q_c_energy_retrieval_prioritizes_q2_over_2c_without_voltage():
    q = "A capacitor has a charge of 20 μC and a capacitance of 5 μF. Calculate the electric field energy stored in the capacitor."
    ctx = retrieve_equations_context(q, formula_top_k=5, example_top_k=5)
    formula_ids = [item.formula.id for item in ctx.selected_formulas]
    example_ids = [item.example.id for item in ctx.selected_examples]
    assert formula_ids[0] == "capacitor_energy_charge_capacitance"
    assert "eq_ex_cap_energy_from_charge_capacitance_nl339" in example_ids

def test_q_c_energy_canonical_fallback_when_llm_emits_charge_voltage_formula():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Q1", "type": "charge", "role": "given", "value": "20", "unit": "uC"},
            {"id": "C1", "type": "capacitance", "role": "given", "value": "5", "unit": "uF"},
            {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "uJ"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_charge_voltage", "objects": ["Q1", "C1", "W_query"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(40.0)
