from __future__ import annotations

import json
import math
import re
from pathlib import Path

from xai_physics.domains.equations.solver import solve_schema


DEFAULT_DATASET = (
    Path(__file__).parent
    / "datasets"
    / "equations_gold.jsonl"
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


def extract_number(answer: str | None) -> float | None:
    if answer is None:
        return None

    match = re.search(
        r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
        str(answer),
    )
    if not match:
        return None

    return float(match.group(0))


def run_benchmark(path: Path = DEFAULT_DATASET) -> dict:
    rows = load_jsonl(path)

    total = len(rows)
    correct = 0
    failures: list[dict] = []

    print(f"Running equations gold-schema benchmark: {path}")
    print("-" * 80)

    for row in rows:
        case_id = row["id"]
        schema = row["schema"]
        expected_value = float(row["expected_value"])
        expected_unit = row.get("expected_unit", "")
        rel_tol = float(row.get("rel_tol", 1e-6))
        abs_tol = float(row.get("abs_tol", 1e-9))

        result = solve_schema(schema)
        predicted = result.answer
        predicted_value = extract_number(predicted)

        ok = (
            result.status in {"ok", "solved"}
            and predicted_value is not None
            and math.isclose(
                predicted_value,
                expected_value,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        )

        if ok:
            correct += 1
            mark = "PASS"
        else:
            mark = "FAIL"
            failures.append(
                {
                    "id": case_id,
                    "expected": f"{expected_value} {expected_unit}".strip(),
                    "predicted": predicted,
                    "predicted_value": predicted_value,
                    "status": result.status,
                    "error": result.error,
                }
            )

        print(
            f"{mark} {case_id}: "
            f"expected={expected_value!r} {expected_unit}, "
            f"predicted={predicted!r}"
        )

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
        help="Path to equations gold JSONL dataset.",
    )
    args = parser.parse_args()

    report = run_benchmark(args.dataset)
    if report["correct"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
