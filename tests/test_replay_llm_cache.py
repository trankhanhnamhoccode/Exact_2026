from __future__ import annotations

import csv
import json

from xai_physics.eval.replay_llm_dataset import compare_answer, run_replay_benchmark
from xai_physics.llm.replay_cache import ReplaySchemaLLM, load_replay_cache, parse_eval_log, write_replay_cache
from xai_physics.llm.schema_pipeline import solve_problem_with_llm


TD009_LOG = """
====================================================================================================
[10/500] row=109 id=TD009

QUESTION:
A parallel-plate capacitor has circular plates with a radius of 10 cm. The distance between the plates and the potential difference across them are 1 cm and 108 V, respectively. The space between the plates is air. What is the charge on the capacitor?

RESULT:
status: solve_failed
domain: equations
answer: None
expected: 3
expected_unit: nC
correct: False
elapsed_sec: 9.363

SCHEMA:
{
  "domain": "equations",
  "objects": [
    {"id": "C1", "type": "capacitance", "role": "constant", "value": null, "unit": "uF"},
    {"id": "U1", "type": "voltage", "role": "given", "value": "108", "unit": "V"},
    {"id": "Q_query", "type": "charge", "role": "query", "value": null, "unit": "uC"}
  ],
  "relations": [
    {"type": "formula", "name": "capacitor_charge_voltage", "objects": ["Q_query", "C1", "U1"]}
  ],
  "constraints": []
}

RUNNING SUMMARY:
solved: 9 | failed: 1
"""


def test_parse_eval_log_to_replay_cache_entries():
    entries = parse_eval_log(TD009_LOG)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.case_id == "TD009"
    assert entry.row == 109
    assert "radius of 10 cm" in entry.question
    assert entry.schema["domain"] == "equations"
    assert entry.metadata == {
        "status": "solve_failed",
        "domain": "equations",
        "answer": "None",
        "expected": "3",
        "expected_unit": "nC",
        "correct": False,
        "elapsed_sec": 9.363,
    }


def test_replay_llm_returns_cached_schema_and_pipeline_can_repair(tmp_path):
    entries = parse_eval_log(TD009_LOG)
    cache_path = tmp_path / "cache.jsonl"
    write_replay_cache(entries, cache_path)
    cache = load_replay_cache(cache_path)
    client = ReplaySchemaLLM(cache)

    client.set_case_id("TD009")
    output = solve_problem_with_llm(entries[0].question, client, k=2)

    assert output.solve_result.status == "ok", output.solve_result.error
    assert output.schema is not None
    assert output.schema["relations"][0]["name"] == "parallel_plate_charge_from_voltage"
    assert "3.004" in str(output.solve_result.answer)


def test_replay_dataset_runner_uses_cache_by_case_id(tmp_path):
    entries = parse_eval_log(TD009_LOG)
    cache_path = tmp_path / "cache.jsonl"
    write_replay_cache(entries, cache_path)

    dataset_path = tmp_path / "dataset.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "unit"])
        writer.writeheader()
        writer.writerow({"id": "TD009", "question": entries[0].question, "answer": "3", "unit": "nC"})

    report = run_replay_benchmark(dataset=dataset_path, cache_path=cache_path, k=2)

    assert report["total"] == 1
    assert report["solved"] == 1
    assert report["correct"] == 1
    assert report["results"][0]["schema_source"] == "formula_driven"


def test_replay_compare_answer_handles_units_and_dirty_scientific_notation():
    from xai_physics.eval.replay_llm_dataset import compare_answer

    assert compare_answer("5006.92525189 pF", "5", expected_unit="nF") is True
    assert compare_answer("0.005231928 uC", "5.23", expected_unit="nC") is True
    assert compare_answer("4.5e+06 V/m", "45.10^{5}", expected_unit="V/m") is True
    assert compare_answer({"Ex": "11 V/m", "magnitude": "245.909 V/m"}, "245.91", expected_unit="N/C") is True


def test_compare_answer_handles_bare_power_of_ten_expected():
    assert compare_answer("9986.16865811 V/m", "10^4", expected_unit="V/m") is True
    assert compare_answer("1e-11 C", "10^{-11}", expected_unit="C") is True


def test_replay_dataset_runner_reports_trusted_accuracy_with_quality_flags(tmp_path):
    from xai_physics.llm.replay_cache import ReplayCacheEntry
    from xai_physics.eval.replay_llm_dataset import load_quality_flags

    entries = parse_eval_log(TD009_LOG)
    bad_gold_entry = ReplayCacheEntry(
        case_id="TD009_BAD_GOLD",
        question=entries[0].question,
        schema=entries[0].schema,
        raw_output=entries[0].raw_output,
        row=110,
        metadata={},
    )
    cache_path = tmp_path / "cache.jsonl"
    write_replay_cache([entries[0], bad_gold_entry], cache_path)

    dataset_path = tmp_path / "dataset.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "unit"])
        writer.writeheader()
        writer.writerow({"id": "TD009", "question": entries[0].question, "answer": "3", "unit": "nC"})
        writer.writerow(
            {
                "id": "TD009_BAD_GOLD",
                "question": entries[0].question,
                "answer": "999",
                "unit": "nC",
            }
        )

    flags_path = tmp_path / "quality_flags.jsonl"
    flags_path.write_text(
        json.dumps(
            {
                "case_id": "TD009_BAD_GOLD",
                "flag": "gold_wrong_high_confidence",
                "reason": "Deliberately wrong test gold.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    flags = load_quality_flags(flags_path)
    assert flags["TD009_BAD_GOLD"].flag == "gold_wrong_high_confidence"

    report = run_replay_benchmark(dataset=dataset_path, cache_path=cache_path, quality_flags_path=flags_path, k=2)

    assert report["total"] == 2
    assert report["correct"] == 1
    assert report["comparable"] == 2
    assert report["excluded_total"] == 1
    assert report["quality_flag_counts"] == {"gold_wrong_high_confidence": 1}
    assert report["trusted_correct"] == 1
    assert report["trusted_comparable"] == 1
    assert report["trusted_accuracy"] == 1.0
    bad_result = next(item for item in report["results"] if item["case_id"] == "TD009_BAD_GOLD")
    assert bad_result["quality_reason"] == "Deliberately wrong test gold."


def test_compare_answer_handles_fraction_times_pi_expected():
    assert compare_answer("0.785398163397 rad", "1/4 \\pi", expected_unit="rad") is True
