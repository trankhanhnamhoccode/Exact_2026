from __future__ import annotations

from typing import Any

from xai_physics.core.units import UNIT_TO_SI, normalize_unit


class ElectrostaticsSchemaValidationError(ValueError):
    pass


SUPPORTED_QUERIES = {
    "net_force",
}

SUPPORTED_OUTPUTS = {
    "magnitude",
    "x_component",
    "y_component",
    "components",
}

SUPPORTED_GEOMETRY = {
    "EquilateralTriangle",
    "Collinear",
}


def _err(message: str) -> None:
    raise ElectrostaticsSchemaValidationError(message)


def _is_quantity_dict(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and "value" in data
        and "unit" in data
        and isinstance(data["value"], (int, float))
        and isinstance(data["unit"], str)
    )


def _validate_quantity(data: Any, path: str) -> None:
    if not _is_quantity_dict(data):
        _err(f"{path} must be a quantity dict with numeric value and string unit.")

    unit = normalize_unit(data["unit"])
    if unit not in UNIT_TO_SI:
        _err(f"{path}.unit is unsupported: {data['unit']}")


def _validate_points(schema: dict[str, Any]) -> set[str]:
    points = schema.get("points")
    if not isinstance(points, list) or not points:
        _err("schema.points must be a non-empty list.")

    has_geometry = bool(schema.get("geometry"))
    ids: set[str] = set()

    for i, point in enumerate(points):
        path = f"points[{i}]"

        if not isinstance(point, dict):
            _err(f"{path} must be an object.")

        point_id = point.get("id")
        if not isinstance(point_id, str) or not point_id:
            _err(f"{path}.id must be a non-empty string.")

        if point_id in ids:
            _err(f"Duplicate point id: {point_id}")

        ids.add(point_id)

        has_x = "x" in point
        has_y = "y" in point

        if has_x != has_y:
            _err(f"{path} must provide both x and y, or neither.")

        if has_x and has_y:
            _validate_quantity(point.get("x"), f"{path}.x")
            _validate_quantity(point.get("y"), f"{path}.y")
        elif not has_geometry:
            _err(f"{path} has no coordinates and schema.geometry is missing.")

    return ids


def _validate_geometry(schema: dict[str, Any], point_ids: set[str]) -> None:
    geometry = schema.get("geometry", [])
    if geometry is None:
        geometry = []

    if not isinstance(geometry, list):
        _err("schema.geometry must be a list if provided.")

    for i, geom in enumerate(geometry):
        path = f"geometry[{i}]"

        if not isinstance(geom, dict):
            _err(f"{path} must be an object.")

        gtype = geom.get("type")
        if gtype not in SUPPORTED_GEOMETRY:
            _err(f"{path}.type is unsupported: {gtype}")

        points = geom.get("points")
        if not isinstance(points, list) or not all(isinstance(p, str) for p in points):
            _err(f"{path}.points must be a list of point ids.")

        for p in points:
            if p not in point_ids:
                _err(f"{path}.points references unknown point: {p}")

        if gtype == "EquilateralTriangle":
            if len(points) != 3:
                _err(f"{path}.points must contain exactly 3 points.")
            _validate_quantity(geom.get("side"), f"{path}.side")

            orientation = geom.get("orientation", "above")
            if orientation not in {"above", "below"}:
                _err(f"{path}.orientation must be 'above' or 'below'.")

        if gtype == "Collinear":
            order = geom.get("order", points)
            if not isinstance(order, list) or len(order) < 2:
                _err(f"{path}.order must contain at least two points.")

            for p in order:
                if p not in point_ids:
                    _err(f"{path}.order references unknown point: {p}")

            distances = geom.get("distances")
            if not isinstance(distances, list) or not distances:
                _err(f"{path}.distances must be a non-empty list.")

            for j, item in enumerate(distances):
                dpath = f"{path}.distances[{j}]"

                pair = item.get("between")
                if not isinstance(pair, list) or len(pair) != 2:
                    _err(f"{dpath}.between must contain exactly two points.")

                for p in pair:
                    if p not in point_ids:
                        _err(f"{dpath}.between references unknown point: {p}")

                pseudo_q = {"value": item.get("value"), "unit": item.get("unit")}
                _validate_quantity(pseudo_q, dpath)


def _validate_charges(schema: dict[str, Any], point_ids: set[str]) -> set[str]:
    charges = schema.get("charges")
    if not isinstance(charges, list) or not charges:
        _err("schema.charges must be a non-empty list.")

    ids: set[str] = set()

    for i, charge in enumerate(charges):
        path = f"charges[{i}]"

        if not isinstance(charge, dict):
            _err(f"{path} must be an object.")

        charge_id = charge.get("id")
        if not isinstance(charge_id, str) or not charge_id:
            _err(f"{path}.id must be a non-empty string.")

        if charge_id in ids:
            _err(f"Duplicate charge id: {charge_id}")

        ids.add(charge_id)

        _validate_quantity(charge.get("charge"), f"{path}.charge")

        at = charge.get("at")
        if at not in point_ids:
            _err(f"{path}.at references unknown point: {at}")

    return ids


def _validate_queries(schema: dict[str, Any], charge_ids: set[str]) -> None:
    queries = schema.get("queries")
    if not isinstance(queries, list) or not queries:
        _err("schema.queries must be a non-empty list.")

    for i, query in enumerate(queries):
        path = f"queries[{i}]"

        if not isinstance(query, dict):
            _err(f"{path} must be an object.")

        qtype = query.get("type")
        if qtype not in SUPPORTED_QUERIES:
            _err(f"{path}.type is unsupported: {qtype}")

        target = query.get("target")
        if target not in charge_ids:
            _err(f"{path}.target references unknown charge: {target}")

        output = query.get("output", "magnitude")
        if output not in SUPPORTED_OUTPUTS:
            _err(f"{path}.output is unsupported: {output}")

        unit = query.get("unit")
        if unit is not None:
            unit = normalize_unit(str(unit))
            if unit not in UNIT_TO_SI:
                _err(f"{path}.unit is unsupported: {query.get('unit')}")


def validate_schema(schema: dict[str, Any]) -> None:
    if not isinstance(schema, dict):
        _err("Schema must be a dict/object.")

    domain = schema.get("domain")
    if domain != "electrostatics":
        _err(f"schema.domain must be 'electrostatics', got {domain!r}.")

    point_ids = _validate_points(schema)
    _validate_geometry(schema, point_ids)
    charge_ids = _validate_charges(schema, point_ids)
    _validate_queries(schema, charge_ids)
