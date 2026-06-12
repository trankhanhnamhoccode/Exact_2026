from __future__ import annotations

import json
from pathlib import Path

from xai_physics.domains.electrostatics.engine import solve_schema


DEFAULT_DATASET = (
    Path(__file__).parent
    / "datasets"
    / "electrostatics_gold.jsonl"
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

    return rows


def run_benchmark(path: Path = DEFAULT_DATASET) -> dict:
    rows = load_jsonl(path)

    total = len(rows)
    correct = 0
    failures: list[dict] = []

    print(f"Running electrostatics gold-schema benchmark: {path}")
    print("-" * 80)

    for row in rows:
        case_id = row["id"]
        schema = row["schema"]
        expected = row["expected_answer"]

        result = solve_schema(schema)
        predicted = result.answer

        ok = result.status == "solved" and predicted == expected

        if ok:
            correct += 1
            mark = "PASS"
        else:
            mark = "FAIL"
            failures.append(
                {
                    "id": case_id,
                    "expected": expected,
                    "predicted": predicted,
                    "status": result.status,
                    "error": result.error,
                }
            )

        print(f"{mark} {case_id}: expected={expected!r}, predicted={predicted!r}")

        if not ok and result.error:
            print(f"  error: {result.error}")

    accuracy = correct / total if total else 0.0

    print("-" * 80)
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "failures": failures,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to electrostatics gold JSONL dataset.",
    )
    args = parser.parse_args()

    run_benchmark(args.dataset)


if __name__ == "__main__":
    main()
