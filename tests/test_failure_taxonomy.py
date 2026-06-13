from __future__ import annotations

from xai_physics.eval.failure_taxonomy import build_taxonomy_report, classify_failure


def test_failure_taxonomy_marks_stale_unit_compare_as_not_solver_error():
    row = {
        "case_id": "TD011",
        "status": "ok",
        "predicted": "5006.92525189 pF",
        "expected": "5",
        "expected_unit": "nF",
        "correct": False,
        "error": None,
        "schema_source": "formula_driven",
    }

    item = classify_failure(row, "What is the capacitance of the capacitor?")

    assert item is not None
    assert item.category == "stale_eval_compare_or_unit_parse"
    assert item.confidence == "high"


def test_failure_taxonomy_detects_force_vs_field_query_mismatch():
    row = {
        "case_id": "LD127",
        "status": "solved",
        "predicted": "3.2e+06 V/m",
        "expected": "44.34",
        "expected_unit": "N",
        "correct": False,
        "error": None,
        "schema_source": None,
    }

    item = classify_failure(row, "Calculate the net electric force acting on q3.")

    assert item is not None
    assert item.category == "electrostatics_force_vs_field_query_mismatch"
    assert item.confidence == "high"


def test_failure_taxonomy_report_groups_categories():
    replay = {
        "results": [
            {
                "case_id": "TD011",
                "status": "ok",
                "predicted": "5006.92525189 pF",
                "expected": "5",
                "expected_unit": "nF",
                "correct": False,
                "error": None,
                "schema_source": "formula_driven",
            },
            {
                "case_id": "DT007",
                "status": "solve_failed",
                "predicted": None,
                "expected": "a/ \\sqrt{2}",
                "expected_unit": "m",
                "correct": None,
                "error": "geometry[1].distance_from_segment.value must be numeric.",
                "schema_source": None,
            },
        ]
    }
    questions = {
        "TD011": "What is the capacitance of the capacitor?",
        "DT007": "Find the point position in terms of a.",
    }

    report = build_taxonomy_report(replay, questions)

    assert report["total_failures"] == 2
    assert report["by_category"]["stale_eval_compare_or_unit_parse"] == 1
    assert report["by_category"]["symbolic_value_not_supported"] == 1
