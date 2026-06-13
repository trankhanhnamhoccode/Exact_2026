from __future__ import annotations

import json
from typing import Any


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        return "\n".join(lines).strip()

    return stripped


def _find_first_json_object_text(text: str) -> str | None:
    start = -1
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue

        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start : i + 1]

    return None


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """
    Extract the first JSON object from an LLM response.

    Accepts:
    - raw JSON object
    - fenced ```json ... ```
    - text containing one JSON object

    Raises ValueError on parse failure or non-object JSON.
    """
    if raw_text is None:
        raise ValueError("LLM response is None.")

    text = _strip_code_fence(str(raw_text))

    candidates = [text]

    first_object = _find_first_json_object_text(text)
    if first_object is not None and first_object != text:
        candidates.append(first_object)

    last_error: Exception | None = None

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

        if not isinstance(parsed, dict):
            raise ValueError("Extracted JSON must be an object/dict.")

        return parsed

    if last_error is not None:
        raise ValueError(f"Could not parse JSON object from LLM response: {last_error}") from last_error

    raise ValueError("Could not find JSON object in LLM response.")
