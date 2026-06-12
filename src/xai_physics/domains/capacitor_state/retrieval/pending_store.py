from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PENDING_PATH = Path(__file__).parent / "pending_examples.jsonl"


@dataclass
class PendingExample:
    id: str
    domain: str
    status: str
    tags: list[str]
    problem: str
    required_engine_features: list[str]
    notes: str = ""


def load_pending_examples(path: Path = DEFAULT_PENDING_PATH) -> list[PendingExample]:
    rows: list[PendingExample] = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                raw: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid pending JSONL at line {line_no}: {exc}") from exc

            rows.append(
                PendingExample(
                    id=raw["id"],
                    domain=raw["domain"],
                    status=raw["status"],
                    tags=list(raw.get("tags", [])),
                    problem=raw["problem"],
                    required_engine_features=list(raw.get("required_engine_features", [])),
                    notes=raw.get("notes", ""),
                )
            )

    return rows


def filter_pending_by_tag(tag: str) -> list[PendingExample]:
    return [
        ex for ex in load_pending_examples()
        if tag in ex.tags
    ]


def filter_pending_by_feature(feature_keyword: str) -> list[PendingExample]:
    key = feature_keyword.lower()

    return [
        ex for ex in load_pending_examples()
        if any(key in feature.lower() for feature in ex.required_engine_features)
    ]
