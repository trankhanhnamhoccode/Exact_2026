import pytest

from xai_physics.domains.equations.text_extractor import extract_equations_schema_from_text
from xai_physics.domains.equations.solver import solve_schema


def _solve(problem: str):
    schema = extract_equations_schema_from_text(problem)
    assert schema is not None, problem
    result = solve_schema(schema)
    assert result.status == "ok", result.error
    return result


def _num(answer: str) -> float:
    import re
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(answer))
    assert m, answer
    return float(m.group(0))


def test_text_extracts_inductor_energy_from_l_and_i_nl304():
    r = _solve("An inductor has an inductance of 0.2 H, and a current of 3 A flows through it. What is the magnetic field energy (J)?")
    assert _num(r.answer) == pytest.approx(0.9)


def test_text_extracts_current_from_inductor_energy_nl313():
    r = _solve("A coil has an inductance of 0.1 H. What current (A) is required to store 0.2 J of magnetic energy?")
    assert _num(r.answer) == pytest.approx(2.0)


def test_text_extracts_inductance_from_energy_and_current_nl334():
    r = _solve("An inductor has a magnetic energy of 0.2 J when the current is 2 A. What is its inductance (H)?")
    assert _num(r.answer) == pytest.approx(0.1)


def test_text_extracts_max_current_from_max_magnetic_energy_nl342():
    r = _solve("A coil has a maximum magnetic energy of 2 J and an inductance of 0.5 H. What is the maximum current (A) through the coil?")
    assert _num(r.answer) == pytest.approx(2.8284271247, rel=1e-6)


def test_text_extracts_lc_natural_frequency_ddt361():
    r = _solve("Calculate the natural oscillation frequency for an LC circuit with L = 2 mH and C = 50 µF.")
    assert _num(r.answer) == pytest.approx(503.292121, rel=1e-6)


def test_text_extracts_lc_natural_period_ddt362():
    r = _solve("Calculate the natural period of oscillation for a circuit with L = 0.1 H and C = 10⁻⁶ F.")
    assert _num(r.answer) == pytest.approx(0.00198691765, rel=1e-6)


def test_text_extracts_lc_angular_frequency_ddt365():
    r = _solve("L = 4 mH, C = 250 µF. Calculate the angular frequency of oscillation.")
    assert _num(r.answer) == pytest.approx(1000.0)


def test_text_extracts_rlc_power_from_u_z_r_ddt328():
    r = _solve("Calculate the power consumed by a circuit with U = 80 V, Z = 40 Ω, and R = 30 Ω.")
    assert _num(r.answer) == pytest.approx(120.0)
