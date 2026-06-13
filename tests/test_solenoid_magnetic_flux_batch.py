import re

import pytest

from xai_physics.domains.equations.solver import solve_schema
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context


def _num(answer: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)
    assert m, f"No number found in answer: {answer!r}"
    return float(m.group(0))


def test_magnetic_flux_one_turn_computes_b_from_solenoid_data():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "A1", "type": "area", "role": "given", "value": "6", "unit": "cm2"},
            {"id": "n1", "type": "turn_density", "role": "given", "value": "1000", "unit": "turns/m"},
            {"id": "I1", "type": "current", "role": "given", "value": "1.5", "unit": "A"},
            {"id": "Phi_query", "type": "magnetic_flux", "role": "query", "value": None, "unit": "Wb"},
        ],
        "relations": [{"type": "formula", "name": "magnetic_flux", "objects": ["Phi_query", "A1", "n1", "I1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(1.130973355e-6, rel=1e-6)


def test_magnetic_flux_linkage_from_flux_per_turn():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Phi1", "type": "magnetic_flux", "role": "given", "value": "5e-6", "unit": "Wb"},
            {"id": "N1", "type": "turn_count", "role": "given", "value": "500", "unit": "turns"},
            {"id": "Lambda_query", "type": "magnetic_flux_linkage", "role": "query", "value": None, "unit": "Wb"},
        ],
        "relations": [{"type": "formula", "name": "magnetic_flux_linkage", "objects": ["Lambda_query", "N1", "Phi1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(0.0025)


def test_magnetic_energy_density_computes_b_from_turn_density_current():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "n1", "type": "turn_density", "role": "given", "value": "1000", "unit": "turns/m"},
            {"id": "I1", "type": "current", "role": "given", "value": "2", "unit": "A"},
            {"id": "u_query", "type": "energy_density", "role": "query", "value": None, "unit": "J/m3"},
        ],
        "relations": [{"type": "formula", "name": "magnetic_energy_density", "objects": ["u_query", "n1", "I1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _num(r.answer) == pytest.approx(2.513274122, rel=1e-6)


def test_solenoid_magnetic_retrieval_prioritizes_flux_not_linkage_for_one_turn():
    q = "A solenoid has a cross-sectional area of 6 cm², a turn density of 1000 turns/m, and an electric current of 1.5 A. Calculate the magnetic flux through one turn."
    ctx = retrieve_equations_context(q, formula_top_k=5, example_top_k=5)
    formula_ids = [item.formula.id for item in ctx.selected_formulas]
    example_ids = [item.example.id for item in ctx.selected_examples]
    assert formula_ids[0] == "magnetic_flux"
    assert "eq_ex_magnetic_flux_one_turn_from_solenoid_ddt204" in example_ids


def test_solenoid_magnetic_retrieval_prioritizes_flux_linkage():
    q = "A solenoid has a magnetic flux of 5×10⁻⁶ Wb per turn and consists of 500 turns. Calculate the total magnetic flux linkage."
    ctx = retrieve_equations_context(q, formula_top_k=5, example_top_k=5)
    formula_ids = [item.formula.id for item in ctx.selected_formulas]
    assert formula_ids[0] == "magnetic_flux_linkage"
