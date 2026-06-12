from pathlib import Path

from xai_physics.eval.benchmark_equations import load_jsonl, run_benchmark


def test_equations_gold_dataset_exists_and_runs():
    path = Path("src/xai_physics/eval/datasets/equations_gold.jsonl")
    rows = load_jsonl(path)

    assert len(rows) >= 10

    report = run_benchmark(path)

    assert report["total"] == len(rows)
    assert report["correct"] == report["total"]
    assert report["failures"] == []
