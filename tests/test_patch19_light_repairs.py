from __future__ import annotations

import math
import re

from xai_physics.domains.electrostatics.engine import solve_schema as solve_electrostatics
from xai_physics.domains.equations.solver import solve_schema as solve_equations
from xai_physics.eval.replay_llm_dataset import compare_answer
from xai_physics.eval.failure_taxonomy import build_taxonomy_report


def _number(text: object) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", str(text), re.I)
    assert match, f"no numeric token in {text!r}"
    return float(match.group(0))


def test_electrostatics_prefers_force_query_after_intermediate_field() -> None:
    schema = {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "geometry": [
            {
                "type": "PairwiseDistances",
                "points": ["A", "B", "C"],
                "distances": [
                    {"between": ["A", "B"], "value": 10, "unit": "cm"},
                    {"between": ["A", "C"], "value": 12, "unit": "cm"},
                    {"between": ["B", "C"], "value": 12, "unit": "cm"},
                ],
                "orientation": "above",
            }
        ],
        "charges": [
            {"id": "q1", "charge": {"value": 6e-6, "unit": "C"}, "at": "A"},
            {"id": "q2", "charge": {"value": -6e-6, "unit": "C"}, "at": "B"},
            {"id": "q3", "charge": {"value": -3e-8, "unit": "C"}, "at": "C"},
        ],
        "queries": [
            {"type": "electric_field", "target": "C", "output": "magnitude", "unit": "V/m"},
            {"type": "net_force", "target": "q3", "output": "magnitude", "unit": "N"},
        ],
    }

    result = solve_electrostatics(schema)

    assert result.status == "solved"
    assert isinstance(result.answer, dict)
    assert set(result.answer) == {"electric_field", "net_force"}
    assert compare_answer(result.answer, "0.094", expected_unit="N") is True
    assert compare_answer(result.answer, "3.125e6", expected_unit="V/m") is True


def test_series_capacitor_field_uses_voltage_division_for_matching_plate() -> None:
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "C1", "role": "given", "type": "capacitance", "unit": "uF", "value": "2"},
            {"id": "C2", "role": "given", "type": "capacitance", "unit": "uF", "value": "3"},
            {"id": "U1", "role": "given", "type": "voltage", "unit": "V", "value": "100"},
            {"id": "d1", "role": "given", "type": "distance", "unit": "mm", "value": "0.5"},
            {"id": "E_query", "role": "query", "type": "electric_field", "unit": "V/m", "value": None},
        ],
        "relations": [{"type": "formula", "name": "parallel_plate_field", "objects": ["E_query", "U1", "d1"]}],
        "formula_candidates": ["parallel_plate_field"],
    }

    result = solve_equations(schema)

    assert result.status == "ok"
    assert math.isclose(result.answer_meta.numeric_value_si, 120000.0, rel_tol=1e-12)
    assert result.answer == "120000 V/m"


def test_capacitor_energy_voltage_infers_parallel_plate_capacitance() -> None:
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "eps1", "role": "given", "type": "relative_permittivity", "unit": None, "value": 3},
            {"id": "d1", "role": "given", "type": "distance", "unit": "mm", "value": 0.5},
            {"id": "S1", "role": "given", "type": "area", "unit": "cm²", "value": 250},
            {"id": "U1", "role": "given", "type": "voltage", "unit": "V", "value": 50},
            {"id": "W_query", "role": "query", "type": "energy", "unit": "uJ", "value": None},
        ],
        "relations": [{"type": "formula", "name": "capacitor_energy_voltage", "objects": ["C1", "U1", "W_query"]}],
        "formula_candidates": ["capacitor_energy_voltage"],
    }

    result = solve_equations(schema)

    assert result.status == "ok"
    expected_si = 0.5 * 8.854187817e-12 * 3 * 0.025 / 0.0005 * 50**2
    assert math.isclose(result.answer_meta.numeric_value_si, expected_si, rel_tol=1e-12)
    assert result.answer.endswith(" uJ")


def test_compare_answer_accepts_near_zero_numeric_residual() -> None:
    assert compare_answer("3.72529e-09 V/m", "0", expected_unit="V/m") is True


def test_failure_taxonomy_skips_quality_flagged_by_default() -> None:
    replay = {
        "results": [
            {
                "case_id": "BAD_GOLD",
                "status": "ok",
                "predicted": "2 N",
                "expected": "6",
                "expected_unit": "N",
                "correct": False,
                "quality_flag": "gold_wrong_high_confidence",
            },
            {
                "case_id": "REAL_FAIL",
                "status": "solved",
                "predicted": "3e6 V/m",
                "expected": "0.09",
                "expected_unit": "N",
                "correct": False,
            },
        ]
    }

    report = build_taxonomy_report(
        replay,
        {
            "BAD_GOLD": "Find the force.",
            "REAL_FAIL": "Calculate the net electric force acting on q3.",
        },
    )

    assert report["skipped_quality_flagged"] == 1
    assert report["total_failures"] == 1
    assert report["items"][0]["case_id"] == "REAL_FAIL"
