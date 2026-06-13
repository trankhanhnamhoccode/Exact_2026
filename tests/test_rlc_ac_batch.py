
from xai_physics.domains.equations.solver import solve_schema
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context


def test_inductive_reactance_from_l_and_f():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "L1", "type": "inductance", "role": "given", "value": "0.2", "unit": "H"},
            {"id": "f1", "type": "frequency", "role": "given", "value": "60", "unit": "Hz"},
            {"id": "XL_query", "type": "inductive_reactance", "role": "query", "value": None, "unit": "ohm"},
        ],
        "relations": [{"type": "formula", "name": "ac_inductive_reactance", "objects": ["XL_query", "L1", "f1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok"
    assert r.answer.startswith("75.398") or r.answer.startswith("75.4")


def test_capacitive_reactance_from_c_and_f():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C1", "type": "capacitance", "role": "given", "value": "50", "unit": "uF"},
            {"id": "f1", "type": "frequency", "role": "given", "value": "60", "unit": "Hz"},
            {"id": "XC_query", "type": "capacitive_reactance", "role": "query", "value": None, "unit": "ohm"},
        ],
        "relations": [{"type": "formula", "name": "ac_capacitive_reactance", "objects": ["XC_query", "C1", "f1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok"
    assert r.answer.startswith("53.05")


def test_rlc_impedance_computes_reactances_from_l_c_f():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "R1", "type": "resistance", "role": "given", "value": "20", "unit": "ohm"},
            {"id": "L1", "type": "inductance", "role": "given", "value": "0.5", "unit": "H"},
            {"id": "C1", "type": "capacitance", "role": "given", "value": "100", "unit": "uF"},
            {"id": "f1", "type": "frequency", "role": "given", "value": "50", "unit": "Hz"},
            {"id": "Z_query", "type": "impedance", "role": "query", "value": None, "unit": "ohm"},
        ],
        "relations": [{"type": "formula", "name": "ac_impedance", "objects": ["Z_query", "R1", "L1", "C1", "f1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok"
    assert r.answer.startswith("126.83") or r.answer.startswith("126.84")


def test_rlc_power_from_voltage_impedance_resistance():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Z1", "type": "impedance", "role": "given", "value": "50", "unit": "ohm"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "30", "unit": "ohm"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "150", "unit": "V"},
            {"id": "P_query", "type": "power", "role": "query", "value": None, "unit": "W"},
        ],
        "relations": [{"type": "formula", "name": "rlc_power_voltage_impedance_resistance", "objects": ["P_query", "U1", "Z1", "R1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok"
    assert r.answer == "270 W"


def test_rlc_characteristic_from_reactances():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "XL1", "type": "inductive_reactance", "role": "given", "value": "70", "unit": "ohm"},
            {"id": "XC1", "type": "capacitive_reactance", "role": "given", "value": "50", "unit": "ohm"},
            {"id": "char_query", "type": "circuit_characteristic", "role": "query", "value": None, "unit": ""},
        ],
        "relations": [{"type": "formula", "name": "rlc_characteristic_from_reactance", "objects": ["char_query", "XL1", "XC1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok"
    assert r.answer == "inductive"


def test_rlc_retrieval_prioritizes_reactance_and_impedance_examples():
    q = "An RLC circuit has a resistance R = 20 Ω, an inductance L = 0.5 H, a capacitance C = 100 μF, and a frequency f = 50 Hz. Calculate the total impedance Z of the circuit."
    ctx = retrieve_equations_context(q, formula_top_k=5, example_top_k=5)
    formula_ids = [x.formula.id for x in ctx.selected_formulas]
    example_ids = [x.example.id for x in ctx.selected_examples]
    assert "ac_impedance" in formula_ids[:3]
    assert any("rlc_impedance" in ex_id for ex_id in example_ids)
