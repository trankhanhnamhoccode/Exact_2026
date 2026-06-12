from __future__ import annotations

from xai_physics.eval.benchmark_capacitor_state import run_benchmark as run_capacitor
from xai_physics.eval.benchmark_electrostatics import run_benchmark as run_electrostatics
from xai_physics.eval.benchmark_equations import run_benchmark as run_equations


def main() -> None:
    print("=" * 80)
    print("XAI Physics Gold-Schema Benchmark")
    print("=" * 80)

    cap_report = run_capacitor()
    print()

    elec_report = run_electrostatics()
    print()

    eq_report = run_equations()
    print()

    total = cap_report["total"] + elec_report["total"] + eq_report["total"]
    correct = cap_report["correct"] + elec_report["correct"] + eq_report["correct"]
    accuracy = correct / total if total else 0.0

    print("=" * 80)
    print("Overall")
    print("=" * 80)
    print(f"Total accuracy: {correct}/{total} = {accuracy:.2%}")

    if correct != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
