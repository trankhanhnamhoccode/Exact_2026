import re

import pytest

from xai_physics.domains.equations.solver import solve_schema
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_parallel_total_current_from_two_lamps():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "12", "unit": "V"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "10", "unit": "ohm"},
            {"id": "R2", "type": "resistance", "role": "given", "value": "30", "unit": "ohm"},
            {"id": "Itotal_query", "type": "total_current", "role": "query", "value": None, "unit": "A"},
        ],
        "relations": [{"type": "formula", "name": "parallel_total_current", "objects": ["Itotal_query", "U1", "R1", "R2"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(1.6)


def test_parallel_branch_current_selects_matching_branch_index():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "24", "unit": "V"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "6", "unit": "ohm"},
            {"id": "R2", "type": "resistance", "role": "given", "value": "12", "unit": "ohm"},
            {"id": "I2_query", "type": "current", "role": "query", "value": None, "unit": "A", "symbol": "I2"},
        ],
        "relations": [{"type": "formula", "name": "parallel_branch_current", "objects": ["I2_query", "U1", "R1", "R2"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(2.0)


def test_parallel_branch_power_single_lamp():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "10", "unit": "V"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "20", "unit": "ohm"},
            {"id": "P1_query", "type": "power", "role": "query", "value": None, "unit": "W"},
        ],
        "relations": [{"type": "formula", "name": "parallel_branch_power", "objects": ["P1_query", "U1", "R1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(5.0)


def test_parallel_total_power_from_three_lamps():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "12", "unit": "V"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "6", "unit": "ohm"},
            {"id": "R2", "type": "resistance", "role": "given", "value": "12", "unit": "ohm"},
            {"id": "R3", "type": "resistance", "role": "given", "value": "24", "unit": "ohm"},
            {"id": "Ptotal_query", "type": "total_power", "role": "query", "value": None, "unit": "W"},
        ],
        "relations": [{"type": "formula", "name": "parallel_total_power", "objects": ["Ptotal_query", "U1", "R1", "R2", "R3"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(42.0)


def test_parallel_total_current_compatible_formula_fallback():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "12", "unit": "V"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "10", "unit": "ohm"},
            {"id": "R2", "type": "resistance", "role": "given", "value": "30", "unit": "ohm"},
            {"id": "Itotal_query", "type": "total_current", "role": "query", "value": None, "unit": "A"},
        ],
        "relations": [{"type": "formula", "name": "ohm_law", "objects": ["Itotal_query", "U1", "R1", "R2"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(1.6)


def test_parallel_retrieval_prioritizes_total_current_example():
    q = "Two lamps with resistances 10 Ω and 30 Ω are connected in parallel to a 12 V source. Calculate the total current supplied by the source."
    ctx = retrieve_equations_context(q, formula_top_k=5, example_top_k=5)
    formula_ids = [item.formula.id for item in ctx.selected_formulas]
    example_ids = [item.example.id for item in ctx.selected_examples]
    assert formula_ids[0] == "parallel_total_current"
    assert "eq_ex_parallel_total_current_thcb066" in example_ids


def test_parallel_retrieval_prioritizes_total_power():
    q = "Three lamps of 6 Ω, 12 Ω, and 24 Ω are connected in parallel to a 12 V source. Calculate the total power consumed."
    ctx = retrieve_equations_context(q, formula_top_k=5, example_top_k=5)
    formula_ids = [item.formula.id for item in ctx.selected_formulas]
    assert formula_ids[0] == "parallel_total_power"
