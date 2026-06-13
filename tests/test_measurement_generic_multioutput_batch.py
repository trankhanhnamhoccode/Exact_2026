import re

import pytest

from xai_physics.domains.equations.solver import solve_schema
from xai_physics.domains.equations.text_extractor import extract_equations_schema_from_text
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context


def _nums(answer: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)]


def test_generic_percent_uncertainty_voltage_schema():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "6.0", "unit": "V"},
            {"id": "dU1", "type": "voltage_uncertainty", "role": "given", "value": "0.2", "unit": "V"},
            {"id": "percent_query", "type": "percent_error", "role": "query", "value": None, "unit": "%"},
        ],
        "relations": [{"type": "formula", "name": "percentage_relative_error", "objects": ["percent_query", "dU1", "U1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _nums(r.answer)[0] == pytest.approx(3.333333333)


def test_text_extractor_measured_length_plus_minus_thcb089():
    schema = extract_equations_schema_from_text("The measured length is 12.0 ± 0.1 cm. Calculate the percentage relative error.")
    assert schema is not None
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _nums(r.answer)[0] == pytest.approx(0.833333333)


def test_text_extractor_temperature_plus_minus_thcb099():
    schema = extract_equations_schema_from_text("A student measured the temperature as 36.5 ± 0.2 °C. Calculate the percentage relative error.")
    assert schema is not None
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _nums(r.answer)[0] == pytest.approx(0.547945205, rel=1e-6)


def test_parallel_all_currents_multi_output_thcb066_schema():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "9", "unit": "V"},
            {"id": "R1", "type": "resistance", "role": "given", "value": "9", "unit": "ohm"},
            {"id": "R2", "type": "resistance", "role": "given", "value": "9", "unit": "ohm"},
            {"id": "I1_query", "type": "current", "role": "query", "value": None, "unit": "A", "symbol": "I1"},
            {"id": "I2_query", "type": "current", "role": "query", "value": None, "unit": "A", "symbol": "I2"},
            {"id": "Itotal_query", "type": "total_current", "role": "query", "value": None, "unit": "A"},
        ],
        "relations": [{"type": "formula", "name": "parallel_all_currents", "objects": ["I1_query", "I2_query", "Itotal_query", "U1", "R1", "R2"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _nums(r.answer) == pytest.approx([1.0, 1.0, 2.0])


def test_text_extractor_parallel_all_currents_thcb066():
    schema = extract_equations_schema_from_text("A voltage source of U = 9V. Two lamps are connected in parallel, and each lamp has a resistance R = 9Ω. Calculate the current through each lamp and the total current.")
    assert schema is not None
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _nums(r.answer) == pytest.approx([1.0, 1.0, 2.0])


def test_text_extractor_total_power_from_voltage_current_thcb075():
    schema = extract_equations_schema_from_text("A 24V source supplies 2 A of current to a circuit with two parallel lamps. Calculate the total power consumption of the circuit.")
    assert schema is not None
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _nums(r.answer)[0] == pytest.approx(48.0)


def test_total_power_sum_lamps_thcb077():
    schema = extract_equations_schema_from_text("The power consumption of lamps D1 and D2 is 10W and 20W, respectively. Calculate the total power of the circuit.")
    assert schema is not None
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    assert _nums(r.answer)[0] == pytest.approx(30.0)


def test_retrieval_prioritizes_generic_measurement_uncertainty():
    ctx = retrieve_equations_context("The measured voltage is 6.0 ± 0.2 V. Calculate the percentage relative error.", formula_top_k=5, example_top_k=5)
    formula_ids = [item.formula.id for item in ctx.selected_formulas]
    assert formula_ids[0] == "percentage_relative_error"


def test_retrieval_prioritizes_parallel_all_currents():
    ctx = retrieve_equations_context("Two lamps are connected in parallel to a 9 V source. Each lamp has resistance 9 ohm. Calculate the current through each lamp and the total current.", formula_top_k=5, example_top_k=5)
    formula_ids = [item.formula.id for item in ctx.selected_formulas]
    assert formula_ids[0] == "parallel_all_currents"
