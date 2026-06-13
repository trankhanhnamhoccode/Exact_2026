
from xai_physics.domains.equations.solver import solve_schema
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context


def _solve(schema):
    r = solve_schema(schema)
    assert r.status == "ok", r.error
    return r.answer


def test_percentage_relative_error_from_least_count():
    ans = _solve({
        "domain": "equations",
        "objects": [
            {"id": "least_count1", "type": "least_count", "role": "given", "value": "0.1", "unit": "cm"},
            {"id": "measured1", "type": "measured_value", "role": "given", "value": "5.0", "unit": "cm"},
            {"id": "percent_query", "type": "percent_error", "role": "query", "value": None, "unit": "%"},
        ],
        "relations": [{"type": "formula", "name": "percentage_relative_error", "objects": ["percent_query", "least_count1", "measured1"]}],
    })
    assert ans == "2 %"


def test_true_measured_absolute_and_relative_errors():
    ans = _solve({
        "domain": "equations",
        "objects": [
            {"id": "actual1", "type": "actual_value", "role": "given", "value": "50.0", "unit": "cm"},
            {"id": "measured1", "type": "measured_value", "role": "given", "value": "49.4", "unit": "cm"},
            {"id": "abs_query", "type": "absolute_error", "role": "query", "value": None, "unit": "cm"},
            {"id": "rel_query", "type": "relative_error", "role": "query", "value": None, "unit": "%"},
        ],
        "relations": [{"type": "formula", "name": "absolute_error_from_actual", "objects": ["abs_query", "rel_query", "actual1", "measured1"]}],
    })
    assert ans.startswith("0.6 cm; 1.214")


def test_random_error_half_range_not_average_absolute_error():
    ans = _solve({
        "domain": "equations",
        "objects": [
            {"id": "i1", "type": "current", "role": "given", "value": "1.8", "unit": "A"},
            {"id": "i2", "type": "current", "role": "given", "value": "2.0", "unit": "A"},
            {"id": "i3", "type": "current", "role": "given", "value": "2.2", "unit": "A"},
            {"id": "err_query", "type": "random_error", "role": "query", "value": None, "unit": "A"},
        ],
        "relations": [{"type": "formula", "name": "random_error", "objects": ["i1", "i2", "i3", "err_query"]}],
    })
    assert ans == "0.2 A"


def test_maximum_possible_value():
    ans = _solve({
        "domain": "equations",
        "objects": [
            {"id": "I1", "type": "measured_value", "role": "given", "value": "0.25", "unit": "A"},
            {"id": "dI1", "type": "absolute_error", "role": "given", "value": "0.01", "unit": "A"},
            {"id": "Imax_query", "type": "maximum_value", "role": "query", "value": None, "unit": "A"},
        ],
        "relations": [{"type": "formula", "name": "measurement_maximum", "objects": ["Imax_query", "I1", "dI1"]}],
    })
    assert ans == "0.26 A"


def test_resistance_uncertainty_quotient():
    ans = _solve({
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "6.0", "unit": "V"},
            {"id": "dU1", "type": "voltage_uncertainty", "role": "given", "value": "0.1", "unit": "V", "symbol": "Delta U"},
            {"id": "I1", "type": "current", "role": "given", "value": "0.3", "unit": "A"},
            {"id": "dI1", "type": "current_uncertainty", "role": "given", "value": "0.01", "unit": "A", "symbol": "Delta I"},
            {"id": "dR_query", "type": "absolute_error", "role": "query", "value": None, "unit": "ohm"},
        ],
        "relations": [{"type": "formula", "name": "resistance_uncertainty_quotient", "objects": ["dR_query", "U1", "dU1", "I1", "dI1"]}],
    })
    assert ans == "1 ohm"


def test_power_relative_and_absolute_uncertainty():
    ans = _solve({
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "9.5", "unit": "V"},
            {"id": "dU1", "type": "voltage_uncertainty", "role": "given", "value": "0.2", "unit": "V"},
            {"id": "I1", "type": "current", "role": "given", "value": "0.95", "unit": "A"},
            {"id": "dI1", "type": "current_uncertainty", "role": "given", "value": "0.02", "unit": "A"},
            {"id": "percent_query", "type": "percent_error", "role": "query", "value": None, "unit": "%"},
        ],
        "relations": [{"type": "formula", "name": "power_uncertainty_product", "objects": ["percent_query", "U1", "dU1", "I1", "dI1"]}],
    })
    assert ans.startswith("4.2105")

    ans2 = _solve({
        "domain": "equations",
        "objects": [
            {"id": "U1", "type": "voltage", "role": "given", "value": "6.3", "unit": "V"},
            {"id": "dU1", "type": "voltage_uncertainty", "role": "given", "value": "0.1", "unit": "V"},
            {"id": "I1", "type": "current", "role": "given", "value": "0.6", "unit": "A"},
            {"id": "dI1", "type": "current_uncertainty", "role": "given", "value": "0.02", "unit": "A"},
            {"id": "dP_query", "type": "absolute_error", "role": "query", "value": None, "unit": "W"},
        ],
        "relations": [{"type": "formula", "name": "power_uncertainty_product", "objects": ["dP_query", "U1", "dU1", "I1", "dI1"]}],
    })
    assert ans2 == "0.186 W"


def test_measurement_retrieval_for_error_propagation_cases():
    q = "Resistance R is calculated using the formula R = U/I, where U = 6.0 ± 0.1 V and I = 0.3 ± 0.01 A. What is the absolute error of R?"
    ctx = retrieve_equations_context(q, formula_top_k=5, example_top_k=5)
    assert "resistance_uncertainty_quotient" in [x.formula.id for x in ctx.selected_formulas]
    assert any(x.example.id == "eq_ex_measurement_resistance_uncertainty_thcb003" for x in ctx.selected_examples)

    q2 = "A student measures the current 3 times and obtains the values: 1.8 A, 2.0 A, 2.2 A. What is the random error?"
    ctx2 = retrieve_equations_context(q2, formula_top_k=5, example_top_k=5)
    assert "random_error_half_range" in [x.formula.id for x in ctx2.selected_formulas]
