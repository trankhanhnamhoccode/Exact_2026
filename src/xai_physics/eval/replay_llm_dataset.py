from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from xai_physics.llm.replay_cache import ReplaySchemaLLM, load_replay_cache
from xai_physics.llm.schema_pipeline import solve_problem_with_llm


_SUPERSCRIPT_TRANS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_FRACTION_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)\s*$")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_SCI_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*(?:x)\s*10\s*([-+]?\d+)",
    re.IGNORECASE,
)

_UNIT_FACTORS = {
    "": 1.0,
    "-": 1.0,
    "—": 1.0,
    "pf": 1e-12,
    "nf": 1e-9,
    "uf": 1e-6,
    "μf": 1e-6,
    "µf": 1e-6,
    "mf": 1e-3,
    "f": 1.0,
    "pc": 1e-12,
    "nc": 1e-9,
    "uc": 1e-6,
    "μc": 1e-6,
    "µc": 1e-6,
    "mc": 1e-3,
    "c": 1.0,
    "pj": 1e-12,
    "nj": 1e-9,
    "uj": 1e-6,
    "μj": 1e-6,
    "µj": 1e-6,
    "mj": 1e-3,
    "j": 1.0,
    "mm": 1e-3,
    "cm": 1e-2,
    "m": 1.0,
    "v/m": 1.0,
    "n/c": 1.0,
    "v": 1.0,
    "n": 1.0,
    "kg": 1.0,
    "rad": 1.0,
    "degree": 1.0,
}


@dataclass(frozen=True)
class ReplayEvalRow:
    case_id: str
    question: str
    expected: str | None
    expected_unit: str | None


@dataclass(frozen=True)
class ReplayEvalResult:
    case_id: str
    status: str
    predicted: str | None
    expected: str | None
    expected_unit: str | None
    correct: bool | None
    error: str | None
    schema_source: str | None


def _parse_case_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def _normalize_math_text(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().translate(_SUPERSCRIPT_TRANS)
    text = text.replace("×", "x").replace("*", "x").replace("·", "x")
    text = text.replace(" . ", " x ").replace(".10", "x10")
    text = text.replace("{", "").replace("}", "").replace("^", "")
    return text


def _first_number(value: str | None) -> float | None:
    raw = "" if value is None else str(value).strip().lower().translate(_SUPERSCRIPT_TRANS)
    raw_compact = raw.replace("{", "").replace("}", "").replace(" ", "")
    bare_power = re.fullmatch(r"10(?:\^|\*\*)?([-+]\d+)", raw_compact) or re.fullmatch(r"10(?:\^|\*\*)(\d+)", raw_compact)
    if bare_power:
        return 10 ** int(bare_power.group(1))

    text = _normalize_math_text(value)
    if not text:
        return None

    frac = _FRACTION_RE.match(text)
    if frac:
        denominator = float(frac.group(2))
        if denominator != 0:
            return float(frac.group(1)) / denominator

    sci = _SCI_RE.search(text)
    if sci:
        return float(sci.group(1)) * (10 ** int(sci.group(2)))

    # Dirty gold strings may contain symbolic forms. For replay benchmarking,
    # use the first numeric token instead of evaluating arbitrary expressions.
    num = _NUMBER_RE.search(text)
    if num:
        return float(num.group(0))
    return None


def _normalize_unit(unit: str | None) -> str:
    if unit is None:
        return ""
    return _normalize_math_text(unit).replace(" ", "")


def _predicted_number_and_unit(value: Any) -> tuple[float | None, str]:
    if isinstance(value, dict):
        value = value.get("magnitude") or next(iter(value.values()), None)
    if value is None:
        return None, ""

    text = _normalize_math_text(str(value))
    num = _NUMBER_RE.search(text)
    if not num:
        return None, ""

    unit = text[num.end():].strip().split()[0] if text[num.end():].strip() else ""
    return float(num.group(0)), _normalize_unit(unit)


def compare_answer(
    predicted: Any,
    expected: str | None,
    *,
    expected_unit: str | None = None,
    rel_tol: float = 5e-2,
    abs_tol: float = 1e-9,
) -> bool | None:
    expected_num = _first_number(expected)
    predicted_num, predicted_unit = _predicted_number_and_unit(predicted)
    if expected_num is None or predicted_num is None:
        return None

    expected_unit_norm = _normalize_unit(expected_unit)
    if predicted_unit in _UNIT_FACTORS and expected_unit_norm in _UNIT_FACTORS:
        expected_factor = _UNIT_FACTORS[expected_unit_norm]
        if expected_factor != 0:
            predicted_num = predicted_num * _UNIT_FACTORS[predicted_unit] / expected_factor

    return math.isclose(predicted_num, expected_num, rel_tol=rel_tol, abs_tol=abs_tol)


def load_dataset_rows(
    path: Path,
    *,
    id_column: str = "id",
    question_column: str = "question",
    expected_column: str = "answer",
    unit_column: str = "unit",
    start: int = 0,
    limit: int | None = None,
    case_ids: set[str] | None = None,
) -> list[ReplayEvalRow]:
    rows: list[ReplayEvalRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < start:
                continue
            case_id = str(row.get(id_column) or "").strip()
            if not case_id:
                continue
            if case_ids is not None and case_id not in case_ids:
                continue
            rows.append(
                ReplayEvalRow(
                    case_id=case_id,
                    question=str(row.get(question_column) or ""),
                    expected=row.get(expected_column),
                    expected_unit=row.get(unit_column),
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def run_replay_benchmark(
    *,
    dataset: Path,
    cache_path: Path,
    k: int = 2,
    start: int = 0,
    limit: int | None = None,
    case_ids: set[str] | None = None,
    skip_missing_cache: bool = False,
    id_column: str = "id",
    question_column: str = "question",
    expected_column: str = "answer",
    unit_column: str = "unit",
) -> dict[str, Any]:
    cache = load_replay_cache(cache_path)
    client = ReplaySchemaLLM(cache)
    rows = load_dataset_rows(
        dataset,
        id_column=id_column,
        question_column=question_column,
        expected_column=expected_column,
        unit_column=unit_column,
        start=start,
        limit=limit,
        case_ids=case_ids,
    )

    results: list[ReplayEvalResult] = []
    for row in rows:
        if row.case_id not in cache:
            if skip_missing_cache:
                continue
            results.append(
                ReplayEvalResult(
                    case_id=row.case_id,
                    status="missing_cache",
                    predicted=None,
                    expected=row.expected,
                    expected_unit=row.expected_unit,
                    correct=False,
                    error=f"No cached schema for {row.case_id}",
                    schema_source=None,
                )
            )
            continue

        try:
            client.set_case_id(row.case_id)
            output = solve_problem_with_llm(row.question, client, k=k)
            result = output.solve_result
            schema_source = None
            for step in result.trace:
                data = getattr(step, "data", None)
                if isinstance(data, dict) and "candidate_source" in data:
                    schema_source = str(data["candidate_source"])
            correct = compare_answer(result.answer, row.expected, expected_unit=row.expected_unit)
            results.append(
                ReplayEvalResult(
                    case_id=row.case_id,
                    status=result.status,
                    predicted=result.answer,
                    expected=row.expected,
                    expected_unit=row.expected_unit,
                    correct=correct,
                    error=result.error,
                    schema_source=schema_source,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive CLI boundary
            results.append(
                ReplayEvalResult(
                    case_id=row.case_id,
                    status="exception",
                    predicted=None,
                    expected=row.expected,
                    expected_unit=row.expected_unit,
                    correct=False,
                    error=str(exc),
                    schema_source=None,
                )
            )

    total = len(results)
    comparable = [r for r in results if r.correct is not None]
    correct_count = sum(1 for r in comparable if r.correct)
    solved_count = sum(1 for r in results if r.status in {"ok", "solved"})

    return {
        "total": total,
        "solved": solved_count,
        "comparable": len(comparable),
        "correct": correct_count,
        "accuracy": correct_count / len(comparable) if comparable else 0.0,
        "results": [r.__dict__ for r in results],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay cached LLM schemas through the normal schema pipeline.")
    parser.add_argument("--dataset", type=Path, required=True, help="CSV dataset path.")
    parser.add_argument("--cache", type=Path, required=True, help="Replay cache JSONL created from an LLM eval log.")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", type=str, help="Comma-separated case ids to replay.")
    parser.add_argument("--skip-missing-cache", action="store_true")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--expected-column", default="answer")
    parser.add_argument("--unit-column", default="unit")
    parser.add_argument("--json-out", type=Path, help="Optional path to save detailed JSON results.")
    args = parser.parse_args(argv)

    report = run_replay_benchmark(
        dataset=args.dataset,
        cache_path=args.cache,
        k=args.k,
        start=args.start,
        limit=args.limit,
        case_ids=_parse_case_ids(args.case_ids),
        skip_missing_cache=args.skip_missing_cache,
        id_column=args.id_column,
        question_column=args.question_column,
        expected_column=args.expected_column,
        unit_column=args.unit_column,
    )

    print(f"replayed: {report['total']}")
    print(f"solved: {report['solved']}")
    print(f"correct: {report['correct']} / {report['comparable']} = {report['accuracy']:.2%}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote: {args.json_out}")


if __name__ == "__main__":
    main()
