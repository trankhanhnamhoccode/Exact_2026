from __future__ import annotations

from typing import Any

from xai_physics.core.units import UNIT_TO_SI, normalize_unit


class SchemaValidationError(ValueError):
    pass


SUPPORTED_EVENTS = {
    "ConnectToSource",
    "DisconnectFromSource",
    "InsertDielectric",
    "DistanceScale",
    "AreaScale",
    "ParallelRedistribution",
    "ShortCircuit",
    "ConnectToInductor",
    "ReplaceDielectric",
    "ReplaceCapacitor",
    "SetCapacitance",
}

SUPPORTED_QUERIES = {
    "voltage",
    "charge",
    "capacitance",
    "energy",
    "energy_ratio",
    "capacitance_ratio",
    "energy_percent",
    "energy_change",
    "energy_reduction",
}


def _err(message: str) -> None:
    raise SchemaValidationError(message)


def _is_quantity_dict(data: Any) -> bool:
    if data is None:
        return False
    if not isinstance(data, dict):
        return False
    if not data:
        return False
    value = data.get("value")
    if value is None:
        # Unknown quantities are allowed in capacitor_state schemas; the engine may infer them.
        return True
    if isinstance(value, bool):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    unit = data.get("unit", "")
    return unit is None or isinstance(unit, str)


def _validate_quantity(data: Any, path: str) -> None:
    if data is None or data == {}:
        return

    if not _is_quantity_dict(data):
        _err(f"{path} must be a quantity dict with numeric value and string unit.")

    if data.get("value") is None:
        return

    unit = normalize_unit(data.get("unit", ""))
    data["unit"] = unit
    if unit and unit not in UNIT_TO_SI:
        _err(f"{path}.unit is unsupported: {data.get('unit')}")


def _validate_entities(schema: dict[str, Any]) -> set[str]:
    entities = schema.get("entities")
    if not isinstance(entities, list) or not entities:
        _err("schema.entities must be a non-empty list.")

    ids: set[str] = set()

    for i, ent in enumerate(entities):
        path = f"entities[{i}]"

        if not isinstance(ent, dict):
            _err(f"{path} must be an object.")

        ent_id = ent.get("id")
        if not isinstance(ent_id, str) or not ent_id:
            _err(f"{path}.id must be a non-empty string.")

        if ent_id in ids:
            _err(f"Duplicate entity id: {ent_id}")
        ids.add(ent_id)

        ent_type = ent.get("type", "capacitor")
        if ent_type != "capacitor":
            _err(f"{path}.type is unsupported: {ent_type}")

        _validate_quantity(ent.get("capacitance"), f"{path}.capacitance")
        _validate_quantity(ent.get("voltage"), f"{path}.voltage")
        _validate_quantity(ent.get("charge"), f"{path}.charge")

        if (
            ent.get("capacitance") is None
            and ent.get("voltage") is None
            and ent.get("charge") is None
        ):
            _err(f"{path} must contain at least one of capacitance, voltage, charge.")

        if "connected_to_source" in ent and not isinstance(ent["connected_to_source"], bool):
            _err(f"{path}.connected_to_source must be boolean.")

    return ids


def _validate_event(event: dict[str, Any], index: int, entity_ids: set[str]) -> None:
    path = f"events[{index}]"

    if not isinstance(event, dict):
        _err(f"{path} must be an object.")

    event_type = event.get("type")
    if event_type not in SUPPORTED_EVENTS:
        _err(f"{path}.type is unsupported: {event_type}")

    apply_to = event.get("apply_to") or event.get("target")
    if apply_to is None:
        _err(f"{path}.apply_to is required.")

    if isinstance(apply_to, str):
        apply_list = [apply_to]
    elif isinstance(apply_to, list) and all(isinstance(x, str) for x in apply_to):
        apply_list = apply_to
    else:
        _err(f"{path}.apply_to must be a string or list of strings.")

    for cap_id in apply_list:
        if cap_id not in entity_ids:
            _err(f"{path}.apply_to references unknown capacitor: {cap_id}")

    params = event.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        _err(f"{path}.params must be an object if provided.")

    if event_type == "ParallelRedistribution":
        if len(apply_list) < 2:
            _err(f"{path}.apply_to must contain at least two capacitors.")

        polarity = params.get("polarity", "same")
        if polarity != "same":
            _err(
                f"{path}.params.polarity={polarity!r} is not supported yet. "
                "Only same-polarity connection is currently supported."
            )

    if event_type == "ConnectToSource":
        _validate_quantity(params.get("voltage"), f"{path}.params.voltage")
        if params.get("voltage") is None:
            _err(f"{path}.params.voltage is required.")

    if event_type == "InsertDielectric":
        k = params.get("dielectric_constant", params.get("k"))
        if not isinstance(k, (int, float)) or k <= 0:
            _err(f"{path}.params.dielectric_constant must be a positive number.")

    if event_type in {"ReplaceCapacitor", "SetCapacitance"}:
        new_cap = (
            params.get("new_capacitance")
            or params.get("capacitance")
            or params.get("final_capacitance")
        )
        _validate_quantity(new_cap, f"{path}.params.new_capacitance")
        if new_cap is None or new_cap == {} or new_cap.get("value") is None:
            _err(f"{path}.params.new_capacitance is required.")

        hold = params.get("hold") or params.get("hold_policy") or params.get("voltage_policy")
        if hold is not None and not isinstance(hold, str):
            _err(f"{path}.params.hold must be a string if provided.")

    if event_type in {"DistanceScale", "AreaScale"}:
        factor = params.get("factor")
        if not isinstance(factor, (int, float)) or factor <= 0:
            _err(f"{path}.params.factor must be a positive number.")


def _validate_query(query: dict[str, Any], index: int, entity_ids: set[str]) -> None:
    path = f"queries[{index}]"

    if not isinstance(query, dict):
        _err(f"{path} must be an object.")

    qtype = query.get("type")
    if qtype not in SUPPORTED_QUERIES:
        _err(f"{path}.type is unsupported: {qtype}")

    target = query.get("target", "system")
    if target != "system" and target not in entity_ids:
        _err(f"{path}.target references unknown capacitor: {target}")

    unit = query.get("unit")
    if unit is not None:
        unit = normalize_unit(str(unit))
        if unit not in UNIT_TO_SI:
            _err(f"{path}.unit is unsupported: {query.get('unit')}")


def validate_schema(schema: dict[str, Any]) -> None:
    if not isinstance(schema, dict):
        _err("Schema must be a dict/object.")

    domain = schema.get("domain")
    if domain != "capacitor_state":
        _err(f"schema.domain must be 'capacitor_state', got {domain!r}.")

    entity_ids = _validate_entities(schema)

    events = schema.get("events", [])
    if not isinstance(events, list):
        _err("schema.events must be a list.")

    for i, event in enumerate(events):
        _validate_event(event, i, entity_ids)

    queries = schema.get("queries")
    if not isinstance(queries, list) or not queries:
        _err("schema.queries must be a non-empty list.")

    for i, query in enumerate(queries):
        _validate_query(query, i, entity_ids)







