from __future__ import annotations

import copy
import re
from typing import Any

_FORCE_HINTS = (
    "force acting",
    "net force",
    "electric force",
    "force exerted",
    "magnitude of the force",
    "force vector",
)
_FIELD_HINTS = (
    "electric field",
    "field strength",
    "field intensity",
    "resultant electric field",
)


def _is_force_question(problem: str) -> bool:
    low = problem.lower()
    return any(hint in low for hint in _FORCE_HINTS) and not any(hint in low for hint in _FIELD_HINTS)


def _target_charge_from_text(problem: str, schema: dict[str, Any]) -> str | None:
    charge_ids = {str(ch.get("id")) for ch in schema.get("charges", []) if isinstance(ch, dict) and ch.get("id")}
    if not charge_ids:
        return None

    patterns = [
        r"(?:acting|exerted)\s+on\s+(?:the\s+)?(?:charge\s+)?(q\d+|q0|q′|q'|qo|q)\b",
        r"(?:force|force\s+vector)\s+(?:acting\s+)?on\s+(?:the\s+)?(?:charge\s+)?(q\d+|q0|q′|q'|qo|q)\b",
        r"(?:charge|electric\s+charge)\s+(q\d+|q0|q′|q'|qo|q)\b\D{0,80}?force",
    ]
    for pattern in patterns:
        m = re.search(pattern, problem, flags=re.I)
        if not m:
            continue
        cid = m.group(1).lower().replace("'", "′")
        if cid == "qo":
            cid = "q0"
        if cid in {"q′", "q'"}:
            cid = "q3"
        if cid in charge_ids:
            return cid

    # Common classroom convention: with q1/q2 sources and q3/q0/q as the test
    # charge, force questions target the test charge.
    for preferred in ("q3", "q0", "q"):
        if preferred in charge_ids:
            return preferred
    return None


def _charge_at_point(schema: dict[str, Any], point_id: str) -> str | None:
    matches = [
        str(ch.get("id"))
        for ch in schema.get("charges", [])
        if isinstance(ch, dict) and ch.get("at") == point_id and isinstance(ch.get("id"), str)
    ]
    return matches[0] if len(matches) == 1 else None


def repair_electrostatics_schema_with_question(problem: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Repair high-confidence electrostatics intent slips using the question text.

    This intentionally handles only a narrow bug: the schema asks for electric
    field while the natural-language question asks for force on a known charge.
    It does not use expected answers/units and therefore avoids learning from
    dirty gold labels.
    """

    if schema.get("domain") != "electrostatics" or not _is_force_question(problem):
        return schema

    queries = schema.get("queries")
    if not isinstance(queries, list) or not queries:
        return schema

    query = queries[0]
    if not isinstance(query, dict) or query.get("type") != "electric_field":
        return schema

    repaired = copy.deepcopy(schema)
    repaired_query = repaired["queries"][0]

    target_charge = _target_charge_from_text(problem, repaired)
    if target_charge is None:
        target = repaired_query.get("target")
        if isinstance(target, str):
            target_charge = _charge_at_point(repaired, target)

    if target_charge is None:
        return schema

    repaired_query["type"] = "net_force"
    repaired_query["target"] = target_charge
    repaired_query["unit"] = "N"
    repaired_query.setdefault("output", "magnitude")
    return repaired
