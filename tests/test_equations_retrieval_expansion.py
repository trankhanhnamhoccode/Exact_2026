from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context


def _top_formula(problem: str) -> str:
    result = retrieve_equations_context(problem, formula_top_k=3, example_top_k=2)
    return result.selected_formulas[0].formula.id


def test_retrieve_parallel_plate_field():
    assert _top_formula(
        "Find the electric field between capacitor plates when voltage is 100 V and plate separation is 0.5 mm."
    ) == "parallel_plate_field"


def test_retrieve_capacitor_energy_density():
    assert _top_formula(
        "Calculate the energy density between the plates of a capacitor from voltage and separation."
    ) == "capacitor_energy_density"


def test_retrieve_capacitor_energy_scaling_constant_voltage():
    assert _top_formula(
        "At constant voltage, a capacitor's capacitance is doubled. How many times does the stored energy change?"
    ) == "capacitor_energy_scaling_constant_voltage"


def test_retrieve_inductor_energy():
    assert _top_formula(
        "Calculate the energy stored in an inductor with inductance 0.2 H and current 3 A."
    ) == "inductor_energy"


def test_retrieve_ac_impedance():
    assert _top_formula(
        "Find the impedance of a series AC circuit with resistance R, inductive reactance XL, and capacitive reactance XC."
    ) == "ac_impedance"


def test_retrieve_power_voltage_resistance():
    assert _top_formula(
        "Calculate the power consumed by a resistor when voltage is 80 V and resistance is 40 ohm."
    ) == "power_voltage_resistance"


def test_retrieve_magnetic_flux_total():
    assert _top_formula(
        "Calculate the magnetic flux through a coil with 1000 turns, magnetic field B, and area A."
    ) == "magnetic_flux_total"


def test_retrieve_absolute_error_from_actual():
    assert _top_formula(
        "Find the absolute error if the actual value is 9.8 m and the measured value is 10.0 m."
    ) == "absolute_error_from_actual"


def test_retrieve_lc_electric_energy_time():
    assert _top_formula(
        "In an LC circuit, find the electric energy at time t when total energy and angular frequency omega are given."
    ) == "lc_electric_energy_time"
