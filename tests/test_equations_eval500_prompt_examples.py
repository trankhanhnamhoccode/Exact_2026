from xai_physics.domains.equations.retrieval.example_store import load_examples
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context
from xai_physics.domains.equations.solver import solve_schema


def _ids(problem: str):
    return [item.example.id for item in retrieve_equations_context(problem, formula_top_k=5, example_top_k=5).selected_examples]


def test_eval500_examples_loaded():
    ids = {ex.id for ex in load_examples()}
    assert "eq_ex_cap_electric_field_energy_wording_td359" in ids
    assert "eq_ex_cap_energy_given_charge_voltage_td361" in ids
    assert "eq_ex_cap_voltage_from_desired_energy_td378" in ids
    assert "eq_ex_parallel_plate_dielectric_capacitance_td384" in ids
    assert "eq_ex_measurement_least_count_absolute_error_thcb001" in ids


def test_retrieve_electric_field_energy_as_capacitor_energy_not_density():
    ids = _ids("A capacitor with a capacitance of 4 μF is charged to a voltage of 6 V. Calculate the electric field energy of the capacitor.")
    assert "eq_ex_cap_electric_field_energy_wording_td359" in ids


def test_solve_energy_from_charge_voltage():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Q1", "type": "charge", "role": "given", "value": "40", "unit": "uC"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "8", "unit": "V"},
            {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "uJ"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_energy_charge_voltage", "objects": ["W_query", "Q1", "U1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok"
    assert r.answer == "160 uJ"


def test_solve_distance_halved_capacitance_scaling():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C_initial", "type": "capacitance", "role": "given", "value": "10", "unit": "pF"},
            {"id": "d_ratio", "type": "ratio", "role": "given", "value": "0.5", "unit": "times", "symbol": "d2/d1"},
            {"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "pF"},
        ],
        "relations": [{"type": "formula", "name": "parallel_plate_capacitance_distance_scaling", "objects": ["C_initial", "d_ratio", "C_query"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok"
    assert r.answer == "20 pF"


def test_solve_least_count_absolute_error():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "least_count1", "type": "least_count", "role": "given", "value": "0.1", "unit": "A"},
            {"id": "err_query", "type": "absolute_error", "role": "query", "value": None, "unit": "A"},
        ],
        "relations": [{"type": "formula", "name": "instrument_absolute_error", "objects": ["err_query", "least_count1"]}],
    }
    r = solve_schema(schema)
    assert r.status == "ok"
    assert r.answer == "0.1 A"
