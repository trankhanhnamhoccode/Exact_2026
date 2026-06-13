from __future__ import annotations

from xai_physics.eval.answer_mode_classifier import classify_expected_answer


def test_classifies_numeric_symbolic_direction_and_multi_numeric_expected_answers():
    assert classify_expected_answer("0.045", "J").answer_mode == "numeric"
    assert classify_expected_answer(r"\sqrt{2} × F₀", "N").answer_mode == "symbolic_expr"
    assert classify_expected_answer("E1 = (3/4)E2", "-").answer_mode == "symbolic_relation"
    assert classify_expected_answer("Hướng về phía q₂", "-").answer_mode == "direction"
    assert classify_expected_answer("I_D₁ = 1.0; I_D₂ = 1.0; I_total = 2.0", "A; A; A").answer_mode == "numeric_multi"


def test_classifies_qualitative_and_formula_text_expected_answers():
    assert classify_expected_answer("all energy is entirely stored in the magnetic field of the inductor", "—").answer_mode == "qualitative"
    assert classify_expected_answer("W = 1/2 · L · I²", "-").answer_mode == "symbolic_relation"
    assert classify_expected_answer("Do not change", "").answer_mode == "qualitative"
