from __future__ import annotations

import json
from pathlib import Path

from .types import ExampleDoc


DEFAULT_EXAMPLES_PATH = Path(__file__).with_name("examples.jsonl")


def load_examples(path: Path = DEFAULT_EXAMPLES_PATH) -> list[ExampleDoc]:
    examples: list[ExampleDoc] = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

            examples.append(
                ExampleDoc(
                    id=row["id"],
                    problem=row["problem"],
                    formula_id=row["formula_id"],
                    tags=list(row.get("tags", [])),
                    schema=row["schema"],
                )
            )

    return examples
