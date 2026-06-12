from xai_physics.domains.equations.retrieval.example_store import load_examples
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context


def _example_ids(problem: str):
    result = retrieve_equations_context(problem, formula_top_k=3, example_top_k=3)
    return [item.example.id for item in result.selected_examples]


def test_extended_examples_are_loaded():
    examples = load_examples()
    ids = {ex.id for ex in examples}

    assert "eq_ex_scaling_const_voltage_001" in ids
    assert "eq_ex_parallel_plate_field_001" in ids
    assert "eq_ex_inductor_energy_001" in ids
    assert "eq_ex_ac_impedance_001" in ids
    assert "eq_ex_power_voltage_resistance_001" in ids
    assert "eq_ex_absolute_error_001" in ids
    assert "eq_ex_lc_energy_time_001" in ids


def test_retrieve_scaling_example():
    ids = _example_ids(
        "At constant voltage, a capacitor's capacitance is doubled. How many times does the stored energy change?"
    )

    assert "eq_ex_scaling_const_voltage_001" in ids


def test_retrieve_ac_impedance_example():
    ids = _example_ids(
        "Find the impedance of a series AC circuit with resistance 30 ohm, inductive reactance 50 ohm, and capacitive reactance 10 ohm."
    )

    assert "eq_ex_ac_impedance_001" in ids


def test_retrieve_lc_energy_time_example():
    ids = _example_ids(
        "In an LC circuit, find the electric energy at time t when total energy and angular frequency are given."
    )

    assert "eq_ex_lc_energy_time_001" in ids
