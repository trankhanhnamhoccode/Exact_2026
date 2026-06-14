from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xai_physics.eval.replay_llm_dataset import compare_answer


_PREFIX_RE = re.compile(r"^[A-Z]+")


@dataclass(frozen=True)
class TaxonomyItem:
    case_id: str
    category: str
    confidence: str
    status: str
    schema_source: str | None
    predicted: Any
    expected: str | None
    expected_unit: str | None
    error: str | None
    question: str
    note: str

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "confidence": self.confidence,
            "status": self.status,
            "schema_source": self.schema_source,
            "predicted": self.predicted,
            "expected": self.expected,
            "expected_unit": self.expected_unit,
            "error": self.error,
            "question": self.question,
            "note": self.note,
        }


def _prefix(case_id: str) -> str:
    m = _PREFIX_RE.match(case_id)
    return m.group(0) if m else "UNKNOWN"


def _load_questions(path: Path, *, id_column: str = "id", question_column: str = "question") -> dict[str, str]:
    questions: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = str(row.get(id_column) or "").strip()
            if case_id:
                questions[case_id] = str(row.get(question_column) or "")
    return questions


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values())
    return str(value or "")


def _question_intent(question: str) -> str | None:
    low = question.lower()
    if any(p in low for p in ["force acting", "net force", "electric force", "force exerted", "magnitude of the force"]):
        return "force"
    if any(p in low for p in ["electric field", "field strength", "field intensity", "resultant electric field"]):
        return "electric_field"
    if any(p in low for p in ["calculate the charge", "find the charge", "determine the charge", "charge on each plate", "charge stored"]):
        return "charge"
    if any(p in low for p in ["dielectric constant", "relative permittivity"]):
        return "relative_permittivity"
    if "capacitance" in low:
        return "capacitance"
    if "charge" in low:
        return "charge"
    if "voltage" in low or "potential difference" in low:
        return "voltage"
    if "energy" in low:
        return "energy"
    return None


def _unit_family(unit: str | None) -> str | None:
    u = str(unit or "").strip().lower().replace("μ", "u").replace("µ", "u")
    if u in {"n", "newton", "newtons"}:
        return "force"
    if u in {"v/m", "n/c"}:
        return "electric_field"
    if u in {"pf", "nf", "uf", "mf", "f"}:
        return "capacitance"
    if u in {"pc", "nc", "uc", "mc", "c"}:
        return "charge"
    if u in {"v", "mv", "kv"}:
        return "voltage"
    if u in {"pj", "nj", "uj", "mj", "j"}:
        return "energy"
    if u in {"mm", "cm", "m"}:
        return "length"
    if u in {"", "-", "—", "none"}:
        return None
    return u or None


def _predicted_family(predicted: Any) -> str | None:
    text = _text(predicted).lower().replace("μ", "u").replace("µ", "u")
    if "v/m" in text or "n/c" in text:
        return "electric_field"
    if any(unit in text for unit in ["pf", "nf", "uf", "mf"]):
        return "capacitance"
    if any(unit in text for unit in ["pc", "nc", "uc", "mc"]):
        return "charge"
    if " n" in text or text.endswith("n"):
        return "force"
    if " v" in text or text.endswith("v"):
        return "voltage"
    if any(unit in text for unit in ["pj", "nj", "uj", "mj", " j"]):
        return "energy"
    if any(unit in text for unit in ["mm", "cm", " m"]):
        return "length"
    return None


def classify_failure(row: dict[str, Any], question: str) -> TaxonomyItem | None:
    if row.get("correct") is True:
        return None

    case_id = str(row.get("case_id") or "")
    status = str(row.get("status") or "")
    predicted = row.get("predicted")
    expected = row.get("expected")
    expected_unit = row.get("expected_unit")
    error = row.get("error")
    schema_source = row.get("schema_source")

    # Re-check with the current evaluator. This marks stale replay reports from
    # older compare logic without pretending the solver changed.
    current_compare = compare_answer(predicted, expected, expected_unit=expected_unit)
    if current_compare is True:
        category = "stale_eval_compare_or_unit_parse"
        confidence = "high"
        note = "Current evaluator considers predicted and expected numerically equivalent after unit/notation normalization."
    else:
        intent = _question_intent(question)
        expected_family = _unit_family(expected_unit)
        predicted_family = _predicted_family(predicted)

        if status in {"solve_failed", "exception", "missing_cache"}:
            category, confidence, note = _classify_error(error, intent)
        elif intent == "force" and predicted_family == "electric_field":
            category = "electrostatics_force_vs_field_query_mismatch"
            confidence = "high"
            note = "Question asks for force, but predicted answer is electric field."
        elif intent == "electric_field" and predicted_family == "force":
            category = "electrostatics_field_vs_force_query_mismatch"
            confidence = "high"
            note = "Question asks for electric field, but predicted answer is force."
        elif expected_family and predicted_family and expected_family != predicted_family:
            category = "question_expected_predicted_dimension_conflict"
            confidence = "medium"
            note = f"Expected unit family is {expected_family}, predicted family is {predicted_family}, question intent is {intent}."
        elif expected_family is not None and intent is not None and expected_family != intent:
            category = "suspected_gold_unit_or_answer_mismatch"
            confidence = "medium"
            note = f"Question intent is {intent}, but expected unit family is {expected_family}."
        elif _prefix(case_id) in {"DT"} and _looks_symbolic(expected):
            category = "symbolic_or_inverse_geometry_not_supported"
            confidence = "medium"
            note = "Expected answer appears symbolic or asks for an inverse geometry quantity."
        else:
            category = "numeric_answer_mismatch"
            confidence = "low"
            note = "Predicted and expected are both numeric but do not match within tolerance."

    return TaxonomyItem(
        case_id=case_id,
        category=category,
        confidence=confidence,
        status=status,
        schema_source=schema_source,
        predicted=predicted,
        expected=expected,
        expected_unit=expected_unit,
        error=error,
        question=question,
        note=note,
    )


def _looks_symbolic(value: str | None) -> bool:
    text = str(value or "")
    return any(token in text for token in ["\\", "sqrt", "frac", "a", "q", "E_", "pi", "abs"])


def _classify_error(error: str | None, intent: str | None) -> tuple[str, str, str]:
    err = str(error or "")
    low = err.lower()
    if "no formula candidate solved" in low:
        return "unsupported_or_wrong_equation_formula", "medium", err
    if "value must be numeric" in low:
        return "symbolic_value_not_supported", "medium", err
    if "coordinates" in low or "geometry" in low or "distance" in low or "target must reference" in low:
        return "electrostatics_geometry_schema_error", "medium", err
    if "connecttosource" in low:
        return "capacitor_state_event_schema_error", "medium", err
    if "queries must be" in low or "charges must be" in low or "points must be" in low:
        return "malformed_schema_from_llm", "medium", err
    if intent:
        return f"{intent}_solve_failed", "low", err
    return "solve_failed_other", "low", err


def build_taxonomy_report(
    replay_report: dict[str, Any],
    questions: dict[str, str],
    *,
    include_quality_flagged: bool = False,
) -> dict[str, Any]:
    items: list[TaxonomyItem] = []
    skipped_quality_flagged = 0
    for row in replay_report.get("results", []):
        if row.get("quality_flag") and not include_quality_flagged:
            skipped_quality_flagged += 1
            continue
        case_id = str(row.get("case_id") or "")
        item = classify_failure(row, questions.get(case_id, ""))
        if item is not None:
            items.append(item)

    by_category = Counter(item.category for item in items)
    by_prefix = Counter(_prefix(item.case_id) for item in items)
    by_category_prefix: dict[str, dict[str, int]] = defaultdict(dict)
    for item in items:
        prefix = _prefix(item.case_id)
        by_category_prefix[item.category][prefix] = by_category_prefix[item.category].get(prefix, 0) + 1

    return {
        "total_failures": len(items),
        "skipped_quality_flagged": skipped_quality_flagged,
        "by_category": dict(by_category.most_common()),
        "by_prefix": dict(by_prefix.most_common()),
        "by_category_prefix": dict(by_category_prefix),
        "items": [item.to_json_obj() for item in items],
    }


def write_taxonomy_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "category",
                "confidence",
                "status",
                "schema_source",
                "predicted",
                "expected",
                "expected_unit",
                "error",
                "note",
                "question",
            ],
        )
        writer.writeheader()
        for item in report.get("items", []):
            writer.writerow(item)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Classify replay-eval failures into actionable buckets.")
    parser.add_argument("--replay-json", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--question-column", default="question")
    parser.add_argument(
        "--include-quality-flagged",
        action="store_true",
        help="Include failures that replay marked with a quality flag. By default gold/statement flags are skipped.",
    )
    args = parser.parse_args(argv)

    replay_report = json.loads(args.replay_json.read_text(encoding="utf-8"))
    questions = _load_questions(args.dataset, id_column=args.id_column, question_column=args.question_column)
    report = build_taxonomy_report(
        replay_report,
        questions,
        include_quality_flagged=args.include_quality_flagged,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv_out:
        write_taxonomy_csv(report, args.csv_out)

    print(f"failures: {report['total_failures']}")
    if report.get("skipped_quality_flagged"):
        print(f"skipped quality-flagged: {report['skipped_quality_flagged']}")
    print("top categories:")
    for category, count in list(report["by_category"].items())[:12]:
        print(f"  {count:4d}  {category}")


if __name__ == "__main__":
    main()
