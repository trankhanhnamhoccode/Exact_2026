from __future__ import annotations

import json

from xai_physics.eval.replay_llm_dataset import compare_answer
from xai_physics.llm.schema_pipeline import solve_problem_with_llm


class _FakeClient:
    def generate(self, prompt: str) -> str:
        # Deliberately unhelpful schema; the deterministic formula candidates should win.
        return json.dumps({"domain": "equations", "objects": [], "relations": []})


def _answer(question: str):
    return solve_problem_with_llm(question, _FakeClient()).solve_result.answer


def test_patch24_measurement_uncertainty_repairs():
    cases = [
        (
            "In an experiment, the measured voltage was 9.5 ± 0.2 V, and the measured current was 0.95 ± 0.02 A. What is the relative error in the power?",
            "4.21",
            "%",
        ),
        (
            "When measuring voltage with a voltmeter, the result is 6.3 ± 0.1 V. If this is used to calculate power with a current of 0.6 ± 0.02 A, what is the absolute error of the power?",
            "0.19",
            "W",
        ),
        (
            "In a series circuit, resistance R1 = 10 ± 0.5 Ω, R2 = 20 ± 1 Ω. What is the absolute error of the total resistance?",
            "1.5",
            "Ω",
        ),
        (
            "The instrument has a least count of 0.1 cm. The measured value is 5.0 cm. Calculate the percentage relative error.",
            "1.0",
            "%",
        ),
        (
            "A pressure gauge has a least count of 0.2 atm. It measures a pressure of 2.0 atm. Calculate the percentage relative error.",
            "10.0",
            "%",
        ),
        (
            "Mass measurement result: 200.0 ± 1.0 g. Calculate the percentage relative uncertainty.",
            "0.5",
            "%",
        ),
        (
            "The measured length is 70.0 ± 0.2 cm. Calculate the percentage relative uncertainty.",
            "0.29",
            "%",
        ),
        (
            "A student measured the time as 120.0 ± 0.5 s. Calculate the percentage relative uncertainty.",
            "0.42",
            "%",
        ),
        (
            "The measuring instrument has a least count of 0.1 cm. The measured value is 10.0 cm. Calculate the percentage relative error.",
            "1.0",
            "%",
        ),
    ]
    for question, expected, unit in cases:
        assert compare_answer(_answer(question), expected, expected_unit=unit) is True


def test_patch24_parallel_circuit_numeric_repairs():
    cases = [
        (
            "In a parallel circuit with two lamps, the current through D₁ is 0.4 A, and the total current is 1.0 A. Calculate the current through D₂.",
            "I_D₂ = 0.6",
            "A",
        ),
        (
            "Two lamps are connected in parallel, with a total current of 1.2 A. If lamp D₁ is removed, what will be the total current (given that lamp D₂ draws 0.5 A)?",
            "I_total_new = 0.5",
            "A",
        ),
        (
            "The current through lamp D₁ is 1.2 A, and the current through lamp D₂ is 0.8 A. Calculate the total current.",
            "I_total = 2.0",
            "A",
        ),
        (
            "Two identical lamps are connected in parallel and consume a total of 18W. Calculate the power of each lamp.",
            "P = 9.0",
            "W",
        ),
        (
            "A light bulb consumes 12W of power under a voltage of 6V. Calculate the current through the bulb.",
            "I = 2.0",
            "A",
        ),
    ]
    for question, expected, unit in cases:
        assert compare_answer(_answer(question), expected, expected_unit=unit) is True


def test_patch24_parallel_circuit_qualitative_repairs():
    cases = [
        (
            "In an electrical circuit, if the resistance of branch D₂ decreases, how will the current through D₂ change?",
            "Resistance decreases → current increases.",
        ),
        (
            "If the total current increases when the resistance of the variable resistor is decreased, what happens to the light bulbs?",
            "The lamp shines brighter because the current through it increases.",
        ),
        (
            "If the current through one lamp in a parallel circuit increases, how will the total current change?",
            "Total current increases.",
        ),
        (
            "If two bulbs are connected in parallel, both at the same voltage, how bright will the bulb with lower resistance be?",
            "Brighter because the current is higher.",
        ),
    ]
    for question, expected in cases:
        assert compare_answer(_answer(question), expected, expected_unit="—") is True


def test_patch24_flags_underdetermined_parallel_branch_case():
    with open("data/quality/quality_flags_v19.jsonl", encoding="utf-8") as f:
        assert "THCB085" in f.read()
