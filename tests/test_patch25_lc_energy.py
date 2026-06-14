from __future__ import annotations

from xai_physics.eval.replay_llm_dataset import compare_answer
from xai_physics.hybrid.candidate_ranker import SchemaCandidate, select_best_candidate
from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas


def _solve(question: str):
    candidates = [SchemaCandidate("formula_driven", schema) for schema in generate_equations_candidate_schemas(question)]
    selected = select_best_candidate(question, candidates)
    assert selected is not None
    assert selected.solve_result.status == "ok"
    return selected.solve_result.answer


def test_lc_inductor_energy_halved_current_remaining_energy():
    answer = _solve("An inductor has a magnetic field energy of 2.0 mJ. If the current is halved, what is the remaining energy (mJ)?")
    assert compare_answer(answer, "0.50", expected_unit="mJ") is True


def test_lc_equal_electric_magnetic_energy_current_percent():
    answer = _solve("An LC circuit is oscillating. When the electric field energy equals the magnetic field energy, what percentage of the peak current is the instantaneous current?")
    assert compare_answer(answer, "70.7", expected_unit="%") is True


def test_capacitor_energy_voltage_capacitance_microfarads():
    answer = _solve("What is the capacitance (µF) of a charged capacitor that has 0.36 J of energy and a voltage of 120 V?")
    assert compare_answer(answer, "50", expected_unit="µF") is True


def test_capacitor_voltage_function_max_energy():
    answer = _solve("A capacitor has a capacitance of 10 µF. The voltage changes according to U = 100 sin(1000t). What is the maximum electric field energy (J)?")
    assert compare_answer(answer, "0.05", expected_unit="J") is True


def test_capacitor_charge_amplitude_max_energy():
    answer = _solve("A capacitor has a capacitance of 5 µF, and its charge varies from 0 to 1500 µC. What is the maximum electric field energy (J)?")
    assert compare_answer(answer, "0.225", expected_unit="J") is True


def test_inductor_energy_with_sqrt_current():
    answer = _solve("An inductor with an inductance L = 0.5 H is in an ideal LC circuit. When the current through the inductor is 2√2 A, what is the magnetic field energy?")
    assert compare_answer(answer, "2", expected_unit="J") is True


def test_capacitor_time_dependent_voltage_energy():
    answer = _solve("In an LC circuit, a capacitor has a capacitance of 30 µF, and the voltage at time t is 100cos(1500t). Calculate the electric field energy at t = 0.001 s.")
    assert compare_answer(answer, "0.00075", expected_unit="J") is True


def test_lc_conceptual_voltage_square_proportionality():
    answer = _solve("The electric field energy in a capacitor is directly proportional to which of the following quantities?")
    assert compare_answer(answer, "The square of the voltage (U²)", expected_unit="-") is True
