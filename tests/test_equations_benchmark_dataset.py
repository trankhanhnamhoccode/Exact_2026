from xai_physics.eval.benchmark_equations import DEFAULT_DATASET, load_jsonl, run_benchmark


def test_equations_gold_dataset_exists_and_runs():
    path = DEFAULT_DATASET
    rows = load_jsonl(path)

    assert len(rows) >= 10

    report = run_benchmark(path)

    assert report["total"] == len(rows)
    assert report["correct"] == report["total"]
    assert report["failures"] == []
