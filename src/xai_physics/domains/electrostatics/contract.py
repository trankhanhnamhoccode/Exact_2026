from __future__ import annotations

from typing import Any

from xai_physics.core.units import UNIT_TO_SI, normalize_unit


class ElectrostaticsSchemaValidationError(ValueError):
    pass


SUPPORTED_QUERIES = {
    "net_force",
    "electric_field",
    "resultant_vector",
    "resultant_angle",
    "coulomb_equal_charge",
}

POINT_FREE_QUERIES = {"resultant_vector", "resultant_angle", "coulomb_equal_charge"}
VECTOR_QUERIES = {"resultant_vector", "resultant_angle"}

SUPPORTED_OUTPUTS = {
    "magnitude",
    "x_component",
    "y_component",
    "components",
}

SUPPORTED_GEOMETRY = {
    "EquilateralTriangle",
    "IsoscelesRightTriangle",
    "PairwiseDistances",
    "Collinear",
    "Midpoint",
    "PointOnLine",
    "PerpendicularBisectorPoint",
    "Centroid",
    "FootOfPerpendicular",
    "PerpendicularRaysFromPoint",
}


def _err(message: str) -> None:
    raise ElectrostaticsSchemaValidationError(message)


def _coerce_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _is_length_path(path: str) -> bool:
    p = path.lower()
    return any(k in p for k in [
        ".x", ".y", ".side", ".leg", ".distance", "distances[",
        "distance_from", "distance_from_segment",
    ])


def _is_charge_path(path: str) -> bool:
    p = path.lower()
    return ".charge" in p or p.endswith("charge")


def _is_force_path(path: str) -> bool:
    p = path.lower()
    return ".magnitude" in p or "vectors[" in p or "net_force" in p or "resultant_vector" in p


def _is_field_path(path: str) -> bool:
    return "electric_field" in path.lower()


def _default_si_unit_for_path(path: str) -> str:
    if _is_charge_path(path):
        return "C"
    if _is_length_path(path):
        return "m"
    if _is_force_path(path):
        return "N"
    if _is_field_path(path):
        return "V/m"
    return ""


def _contextual_normalize_unit(unit: Any, path: str) -> str:
    raw = "" if unit is None else str(unit).strip()
    raw_lower = raw.lower()

    # Qwen often emits "c" for cm in geometry, while "c" in charge context means Coulomb.
    if raw_lower == "c":
        if _is_length_path(path):
            return "cm"
        if _is_charge_path(path):
            return "C"

    if raw_lower == "n" and _is_force_path(path):
        return "N"

    if raw_lower == "v/m" and _is_field_path(path):
        return "V/m"
    if raw_lower == "n/c" and _is_field_path(path):
        return "N/C"

    normalized = normalize_unit(raw)
    if normalized in UNIT_TO_SI:
        return normalized

    # Missing / unknown / unitless => assume value is already SI by context.
    fallback = _default_si_unit_for_path(path)
    if fallback:
        return fallback

    return ""


def _is_quantity_dict(data: Any) -> bool:
    return isinstance(data, dict) and "value" in data and _coerce_number(data.get("value")) is not None


def _validate_quantity(data: Any, path: str) -> None:
    if not isinstance(data, dict):
        _err(f"{path} must be a quantity object.")

    value = _coerce_number(data.get("value"))
    if value is None:
        _err(f"{path}.value must be numeric.")

    # Mutate schema so downstream engine receives clean numeric values/units.
    data["value"] = value
    unit = _contextual_normalize_unit(data.get("unit"), path)
    data["unit"] = unit

    if unit and unit not in UNIT_TO_SI:
        _err(f"{path}.unit is unsupported after normalization: {unit!r}")


def _query_types(schema: dict[str, Any]) -> list[str]:
    queries = schema.get("queries")
    if not isinstance(queries, list) or not queries:
        _err("schema.queries must be a non-empty list.")
    out: list[str] = []
    for i, query in enumerate(queries):
        if not isinstance(query, dict):
            _err(f"queries[{i}] must be an object.")
        qtype = query.get("type")
        if qtype not in SUPPORTED_QUERIES:
            _err(f"queries[{i}].type is unsupported: {qtype}")
        out.append(qtype)
    return out


def _validate_points(schema: dict[str, Any], required: bool = True) -> set[str]:
    points = schema.get("points", [])
    if not points:
        if required:
            _err("schema.points must be a non-empty list.")
        return set()

    if not isinstance(points, list):
        _err("schema.points must be a list.")

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
        elif required and not has_geometry:
            _err(f"{path} has no coordinates and schema.geometry is missing.")

    return ids


def _validate_distance_item(item: Any, path: str, point_ids: set[str]) -> None:
    if not isinstance(item, dict):
        _err(f"{path} must be an object.")

    pair = item.get("between")
    if not isinstance(pair, list) or len(pair) != 2:
        _err(f"{path}.between must contain exactly two points.")

    for p in pair:
        if p not in point_ids:
            _err(f"{path}.between references unknown point: {p}")

    _validate_quantity(item, path)


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

        points = geom.get("points", [])
        if points and (not isinstance(points, list) or not all(isinstance(p, str) for p in points)):
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

        elif gtype == "IsoscelesRightTriangle":
            if len(points) != 3:
                _err(f"{path}.points must contain exactly 3 points.")
            if geom.get("right_angle_at") not in point_ids:
                _err(f"{path}.right_angle_at references unknown point.")
            _validate_quantity(geom.get("leg"), f"{path}.leg")
            orientation = geom.get("orientation", "above")
            if orientation not in {"above", "below"}:
                _err(f"{path}.orientation must be 'above' or 'below'.")

        elif gtype in {"PairwiseDistances", "Collinear"}:
            if gtype == "PairwiseDistances" and len(points) < 2:
                _err(f"{path}.points must contain at least 2 points.")

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
                _validate_distance_item(item, f"{path}.distances[{j}]", point_ids)

        elif gtype == "Midpoint":
            point = geom.get("point")
            between = geom.get("between")
            if point not in point_ids:
                _err(f"{path}.point references unknown point: {point}")
            if not isinstance(between, list) or len(between) != 2:
                _err(f"{path}.between must contain exactly two points.")
            for p in between:
                if p not in point_ids:
                    _err(f"{path}.between references unknown point: {p}")

        elif gtype == "PointOnLine":
            for key in ["point", "start", "end"]:
                if geom.get(key) not in point_ids:
                    _err(f"{path}.{key} references unknown point: {geom.get(key)}")
            _validate_quantity(geom.get("distance_from_start"), f"{path}.distance_from_start")
            direction = geom.get("direction", "toward_end")
            if direction not in {"toward_end", "away_from_end"}:
                _err(f"{path}.direction must be toward_end or away_from_end.")

        elif gtype == "PerpendicularBisectorPoint":
            point = geom.get("point")
            between = geom.get("between")
            if point not in point_ids:
                _err(f"{path}.point references unknown point: {point}")
            if not isinstance(between, list) or len(between) != 2:
                _err(f"{path}.between must contain exactly two points.")
            for p in between:
                if p not in point_ids:
                    _err(f"{path}.between references unknown point: {p}")
            _validate_quantity(geom.get("distance_from_segment"), f"{path}.distance_from_segment")
            orientation = geom.get("orientation", "above")
            if orientation not in {"above", "below"}:
                _err(f"{path}.orientation must be 'above' or 'below'.")

        elif gtype == "Centroid":
            point = geom.get("point")
            of_points = geom.get("of")
            if point not in point_ids:
                _err(f"{path}.point references unknown point: {point}")
            if not isinstance(of_points, list) or len(of_points) < 2:
                _err(f"{path}.of must contain at least two points.")
            for p in of_points:
                if p not in point_ids:
                    _err(f"{path}.of references unknown point: {p}")

        elif gtype == "FootOfPerpendicular":
            point = geom.get("point")
            from_id = geom.get("from")
            to_line = geom.get("to_line")
            if point not in point_ids:
                _err(f"{path}.point references unknown point: {point}")
            if from_id not in point_ids:
                _err(f"{path}.from references unknown point: {from_id}")
            if not isinstance(to_line, list) or len(to_line) != 2:
                _err(f"{path}.to_line must contain exactly two points.")
            for p in to_line:
                if p not in point_ids:
                    _err(f"{path}.to_line references unknown point: {p}")

        elif gtype == "PerpendicularRaysFromPoint":
            center = geom.get("center")
            if center not in point_ids:
                _err(f"{path}.center references unknown point: {center}")
            ray_points = geom.get("points")
            distances = geom.get("distances")
            if not isinstance(ray_points, list) or not ray_points:
                _err(f"{path}.points must be a non-empty list.")
            if not isinstance(distances, list) or len(distances) != len(ray_points):
                _err(f"{path}.distances must match {path}.points length.")
            for p in ray_points:
                if p not in point_ids:
                    _err(f"{path}.points references unknown point: {p}")
            for j, distance in enumerate(distances):
                _validate_quantity(distance, f"{path}.distances[{j}]")


def _validate_medium(schema: dict[str, Any]) -> None:
    medium = schema.get("medium")
    if medium is None:
        return
    if not isinstance(medium, dict):
        _err("schema.medium must be an object if provided.")
    eps = medium.get("relative_permittivity", medium.get("epsilon_r", 1.0))
    eps_num = _coerce_number(eps)
    if eps_num is None or eps_num <= 0:
        _err("schema.medium.relative_permittivity must be a positive number.")
    medium["relative_permittivity"] = eps_num


def _validate_charges(schema: dict[str, Any], point_ids: set[str], required: bool = True) -> set[str]:
    charges = schema.get("charges", [])
    if not charges:
        if required:
            _err("schema.charges must be a non-empty list.")
        return set()

    if not isinstance(charges, list):
        _err("schema.charges must be a list.")

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


def _validate_vectors(schema: dict[str, Any], required: bool = False) -> set[str]:
    vectors = schema.get("vectors", [])
    if not vectors:
        if required:
            _err("schema.vectors must be a non-empty list for resultant_vector queries.")
        return set()

    if not isinstance(vectors, list):
        _err("schema.vectors must be a list.")

    ids: set[str] = set()
    for i, vec in enumerate(vectors):
        path = f"vectors[{i}]"
        if not isinstance(vec, dict):
            _err(f"{path} must be an object.")
        vid = vec.get("id")
        if not isinstance(vid, str) or not vid:
            _err(f"{path}.id must be a non-empty string.")
        if vid in ids:
            _err(f"Duplicate vector id: {vid}")
        ids.add(vid)
        _validate_quantity(vec.get("magnitude"), f"{path}.magnitude")
        if "angle_deg" in vec and not isinstance(vec.get("angle_deg"), (int, float)):
            _err(f"{path}.angle_deg must be numeric if provided.")
    return ids


def _validate_queries(
    schema: dict[str, Any],
    point_ids: set[str],
    charge_ids: set[str],
    vector_ids: set[str],
) -> None:
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
        if qtype == "net_force":
            if target not in charge_ids:
                _err(f"{path}.target references unknown charge: {target}")
        elif qtype == "electric_field":
            if target not in point_ids and target not in charge_ids:
                _err(f"{path}.target must reference a known point or charge: {target}")
            exclude = query.get("exclude_sources", [])
            if exclude is not None:
                if not isinstance(exclude, list):
                    _err(f"{path}.exclude_sources must be a list if provided.")
                for cid in exclude:
                    if cid not in charge_ids:
                        _err(f"{path}.exclude_sources references unknown charge: {cid}")
        elif qtype == "resultant_vector":
            if target not in {"vectors", "system", None} and target not in vector_ids:
                _err(f"{path}.target must be 'vectors', 'system', omitted, or a vector id.")
        elif qtype == "resultant_angle":
            if target not in {"vectors", "system", None}:
                _err(f"{path}.target must be 'vectors', 'system', or omitted for resultant_angle.")
            _validate_quantity(query.get("resultant"), f"{path}.resultant")
        elif qtype == "coulomb_equal_charge":
            _validate_quantity(query.get("force"), f"{path}.force")
            _validate_quantity(query.get("distance"), f"{path}.distance")

        output = query.get("output", "magnitude")
        if output not in SUPPORTED_OUTPUTS:
            _err(f"{path}.output is unsupported: {output}")

        unit = query.get("unit")
        if qtype == "electric_field":
            default_query_unit = "V/m"
        elif qtype in {"net_force", "resultant_vector"}:
            default_query_unit = "N"
        else:
            default_query_unit = ""

        if unit is None:
            query["unit"] = default_query_unit
        else:
            normalized = _contextual_normalize_unit(unit, f"{path}.{qtype}.unit")
            query["unit"] = normalized if normalized in UNIT_TO_SI else default_query_unit


def validate_schema(schema: dict[str, Any]) -> None:
    if not isinstance(schema, dict):
        _err("Schema must be a dict/object.")

    domain = schema.get("domain")
    if domain != "electrostatics":
        _err(f"schema.domain must be 'electrostatics', got {domain!r}.")

    qtypes = _query_types(schema)
    point_free = all(qtype in POINT_FREE_QUERIES for qtype in qtypes)
    vector_required = all(qtype in VECTOR_QUERIES for qtype in qtypes)

    point_ids = _validate_points(schema, required=not point_free)
    if point_ids:
        _validate_geometry(schema, point_ids)
    elif schema.get("geometry"):
        _err("schema.geometry requires schema.points.")

    _validate_medium(schema)
    charge_ids = _validate_charges(schema, point_ids, required=not point_free)
    vector_ids = _validate_vectors(schema, required=vector_required)
    _validate_queries(schema, point_ids, charge_ids, vector_ids)
