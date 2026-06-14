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


def test_chlt_resonance_boolean_returns_yes_not_f0():
    answer = _solve("An RLC series circuit has R=75 Ω, L=0.2 H, C=40 μF. Is 56.3 Hz the resonant frequency?")
    assert compare_answer(answer, "Yes", expected_unit="-") is True


def test_inverse_resonance_capacitance_from_l_and_frequency():
    answer = _solve("Calculate C for L = 0.15 H resonating at 40 Hz.")
    assert compare_answer(answer, "105.54", expected_unit="µF") is True


def test_frequency_scaled_rlc_original_inductive_reactance():
    answer = _solve(
        "R = 20Ω. The resonant current (I_resonance) is 4A at f = 60Hz. "
        "When the frequency doubles, the current I becomes 2A. What was the initial ZL?"
    )
    assert compare_answer(answer, "23.09", expected_unit="Ω") is True


def test_waveform_series_rlc_impedance_and_capacitor_voltage():
    impedance = _solve(
        "A voltage u = 200√2 cos 100πt (V) is applied to a series RLC circuit with R = 100 Ω, "
        "L = 1/π H, C = 10⁻⁴/(2π) F. Calculate the total impedance Z of the circuit."
    )
    assert compare_answer(impedance, "141.4", expected_unit="Ω") is True

    uc = _solve(
        "A voltage u = 200√2 cos 100πt (V) is applied to a series RLC circuit with R = 100 Ω, "
        "L = 1/π H, C = 10⁻⁴/(2π) F. Calculate the RMS voltage across the capacitor, U_C."
    )
    assert compare_answer(uc, "282.8", expected_unit="V") is True


def test_ab_quadrature_total_power_and_current():
    power = _solve(
        "Circuit AB consists of segment AM (R1 = 20 Ω in series with capacitor C) and segment MB (R2 = 50 Ω in series with inductor L). "
        "Given that LCω² = 1 and uAM is 90° out of phase with uMB. An RMS voltage U = 120 V is applied across AB. "
        "What is the total power consumed by circuit AB?"
    )
    assert compare_answer(power, "205.71", expected_unit="W") is True

    current = _solve(
        "Circuit AB has R1 = 20 Ω and R2 = 30 Ω. It satisfies the condition LCω² = 1, and the voltage uAM is 90 degrees out of phase with uMB. "
        "An RMS voltage U = 80 V is applied across AB. What is the RMS current in the circuit?"
    )
    assert compare_answer(current, "1.6", expected_unit="A") is True


def test_ab_quadrature_section_voltage_and_power_factor():
    umb = _solve(
        "Consider a circuit segment AB. It consists of two parts in series: segment AM contains R1 = 25 Ω in series with C, "
        "and segment MB contains R2 = 40 Ω in series with L. The condition LCω² = 1 is given. "
        "An RMS voltage U_AB = 80 V is applied across AB. Given that uAM is 90 degrees out of phase with uMB. "
        "Calculate the RMS voltage across MB."
    )
    assert compare_answer(umb, "62.76", expected_unit="V") is True

    pf = _solve("The AB circuit satisfies LCω² = 1. The voltages uAM and uMB are 90° out of phase. Given R1 = 30 Ω and R2 = 40 Ω. Calculate the power factor of the entire circuit.")
    assert compare_answer(pf, "1", expected_unit="-") is True


def test_quality_factor_from_lcr():
    answer = _solve("Given L = 0.20 H, C = 80 µF, R = 50 Ω, determine Q.")
    assert compare_answer(answer, "1.00", expected_unit="") is True
