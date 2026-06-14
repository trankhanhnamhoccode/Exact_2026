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


def test_self_induced_emf_from_inductance_and_current_change():
    answer = _solve(
        "A solenoid has an inductance L = 0.3 H. The current increases uniformly from 0 to 5 A in 0.02 s. "
        "What is the induced electromotive force?"
    )
    assert compare_answer(answer, "75.00", expected_unit="V") is True


def test_induced_emf_from_flux_change():
    answer = _solve(
        "A solenoid has a magnetic flux of 0.004 Wb. If the magnetic flux decreases to 0 in 0.02 s, "
        "what is the average induced electromotive force?"
    )
    assert compare_answer(answer, "0.2", expected_unit="V") is True


def test_self_inductance_from_emf_and_current_change():
    answer = _solve(
        "The current through a solenoid increases uniformly from 0 to 5 A over a period of 0.2 s. "
        "Given that the induced electromotive force is 0.5 V, determine the self-inductance."
    )
    assert compare_answer(answer, "0.02", expected_unit="H") is True


def test_solenoid_magnetic_flux_from_geometry():
    answer = _solve(
        "A solenoid is 0.6 m long, has 1200 turns, and a current of 2 A flows through it. "
        "Calculate the magnetic flux through a cross-sectional area of 4×10⁻⁴ m²."
    )
    assert compare_answer(answer, "2.01e-6", expected_unit="Wb") is True


def test_solenoid_magnetic_energy_from_geometry():
    answer = _solve(
        "A solenoid has a length of 1 m, a cross-sectional area of 5 cm², 1000 turns, and carries a current of 2 A. "
        "Calculate the magnetic field energy."
    )
    assert compare_answer(answer, "1.26e-3", expected_unit="J") is True


def test_chlt_resonance_yes_no():
    yes_answer = _solve("Given L = 0.1 H and C = 10 µF, the circuit operates at f = 159.2 Hz. Does resonance occur?")
    assert compare_answer(yes_answer, "Yes", expected_unit="-") is True

    no_answer = _solve("Given L = 0.1 H and C = 10 µF, the circuit operates at f = 100 Hz. Does resonance occur?")
    assert compare_answer(no_answer, "No", expected_unit="-") is True


def test_natural_period_of_lc_circuit():
    answer = _solve("Calculate the natural period of oscillation for a circuit with L = 0.1 H and C = 10⁻⁶ F.")
    assert compare_answer(answer, "1.99e-3", expected_unit="s") is True


def test_solenoid_qualitative_dependence():
    answer = _solve("What quantity does the magnetic field inside a solenoid depend linearly on?")
    assert compare_answer(answer, "Current through the solenoid", expected_unit="—") is True
