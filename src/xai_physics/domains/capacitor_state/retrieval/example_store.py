from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xai_physics.domains.capacitor_state.retrieval.types import SchemaExample


DEFAULT_EXAMPLE_PATH = Path(__file__).parent / "examples.jsonl"


def load_examples(path: Path = DEFAULT_EXAMPLE_PATH) -> list[SchemaExample]:
    examples: list[SchemaExample] = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                raw: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid capacitor example JSONL at line {line_no}: {exc}") from exc

            examples.append(
                SchemaExample(
                    id=raw["id"],
                    domain=raw["domain"],
                    tags=list(raw.get("tags", [])),
                    problem=raw["problem"],
                    schema=raw["schema"],
                )
            )

    return examples
