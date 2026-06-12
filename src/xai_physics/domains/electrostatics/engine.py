from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

from xai_physics.core.result import SolveResult
from xai_physics.core.units import to_si, convert
from xai_physics.domains.electrostatics.contract import validate_schema
from xai_physics.domains.electrostatics.vector import Vec2
from xai_physics.domains.electrostatics.coordinate_builder import build_coordinates


K_COULOMB = 9.0e9


@dataclass
class Point:
    id: str
    position_m: Vec2


@dataclass
class Charge:
    id: str
    charge_C: float
    point_id: str


@dataclass
class VectorInput:
    id: str
    value: Vec2
    unit: str


def _quantity_to_si(data: dict[str, Any]) -> float:
    return to_si(float(data["value"]), str(data["unit"]))


def _point_ids(schema: dict[str, Any]) -> set[str]:
    return {p.get("id") for p in schema.get("points", []) if isinstance(p, dict) and isinstance(p.get("id"), str)}


def _charge_point_map(schema: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for ch in schema.get("charges", []) or []:
        if isinstance(ch, dict) and isinstance(ch.get("id"), str) and isinstance(ch.get("at"), str):
            out[ch["id"]] = ch["at"]
    return out


def _derived_focus_points(schema: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for geom in schema.get("geometry", []) or []:
        if not isinstance(geom, dict):
            continue
        if geom.get("type") in {"Midpoint", "PerpendicularBisectorPoint", "Centroid", "FootOfPerpendicular"}:
            point = geom.get("point")
            if isinstance(point, str) and point not in out:
                out.append(point)
    return out


def _repair_schema_before_validation(schema: dict[str, Any]) -> None:
    """Fix common LLM/extractor slips without changing physics.

    - Midpoint point=m/h but query/target is M/H.
    - Midpoint point accidentally equals endpoint A/B.
    - Electric-field target is a source point while a derived measurement point exists.
    """
    point_ids = _point_ids(schema)
    charge_at = _charge_point_map(schema)
    charge_points = set(charge_at.values())

    query_point_targets: list[str] = []
    for query in schema.get("queries", []) or []:
        if not isinstance(query, dict):
            continue
        qtype = query.get("type")
        target = query.get("target")
        if qtype == "electric_field" and isinstance(target, str) and target in point_ids:
            query_point_targets.append(target)
        elif qtype == "net_force" and isinstance(target, str) and target in charge_at:
            query_point_targets.append(charge_at[target])

    # Repair midpoint target names.
    for geom in schema.get("geometry", []) or []:
        if not isinstance(geom, dict) or geom.get("type") != "Midpoint":
            continue
        between = geom.get("between") or []
        current = geom.get("point")
        desired = [p for p in query_point_targets if p in point_ids and p not in between]
        if not desired:
            continue
        wanted = desired[0]
        if current not in point_ids or current in between or (isinstance(current, str) and current.islower()):
            geom["point"] = wanted

    # Repair electric-field target when it points at a source charge point but
    # the problem also has a clear derived measurement point M/H/O.
    focus = _derived_focus_points(schema)
    non_charge_focus = [p for p in focus if p in point_ids and p not in charge_points]
    # Prefer conventional uppercase measurement points.
    non_charge_focus.sort(key=lambda p: (not p.isupper(), p))

    for query in schema.get("queries", []) or []:
        if not isinstance(query, dict) or query.get("type") != "electric_field":
            continue
        target = query.get("target")
        if isinstance(target, str) and target in charge_points and non_charge_focus:
            query["target"] = non_charge_focus[0]


def _build_points(schema: dict[str, Any]) -> dict[str, Point]:
    coords = build_coordinates(schema)
    return {point_id: Point(id=point_id, position_m=pos) for point_id, pos in coords.items()}


def _build_charges(schema: dict[str, Any]) -> dict[str, Charge]:
    charges: dict[str, Charge] = {}

    for item in schema.get("charges", []) or []:
        charge = Charge(
            id=item["id"],
            charge_C=_quantity_to_si(item["charge"]),
            point_id=item["at"],
        )
        charges[charge.id] = charge

    return charges


def _relative_permittivity(schema: dict[str, Any]) -> float:
    medium = schema.get("medium") or {}
    if not isinstance(medium, dict):
        return 1.0
    return float(medium.get("relative_permittivity", medium.get("epsilon_r", 1.0)))


def _k_eff(schema: dict[str, Any]) -> float:
    return K_COULOMB / _relative_permittivity(schema)


def _net_force_on(
    target_id: str,
    points: dict[str, Point],
    charges: dict[str, Charge],
    k_eff: float,
) -> Vec2:
    if target_id not in charges:
        raise ValueError(f"Unknown target charge: {target_id}")

    target = charges[target_id]
    target_pos = points[target.point_id].position_m

    total = Vec2(0.0, 0.0)

    for source_id, source in charges.items():
        if source_id == target_id:
            continue

        source_pos = points[source.point_id].position_m

        # Vector from source charge to target charge.
        r_vec = target_pos - source_pos
        r = r_vec.norm()

        if r == 0:
            raise ValueError(f"Charges {target_id} and {source_id} occupy the same point.")

        # Coulomb vector force on target due to source:
        # F = k q_target q_source (r_target - r_source) / |r|^3
        scale = k_eff * target.charge_C * source.charge_C / (r**3)
        total = total + r_vec.scale(scale)

    return total


def _target_point_and_exclusions(
    query: dict[str, Any],
    points: dict[str, Point],
    charges: dict[str, Charge],
) -> tuple[str, set[str]]:
    target = query.get("target")
    exclude_sources = set(query.get("exclude_sources", []) or [])

    if target in charges:
        charge = charges[target]
        exclude_sources.add(target)
        return charge.point_id, exclude_sources

    if target in points:
        # Electric field at a point occupied by a source charge usually means the
        # external field at that location, so exclude same-location charges.
        for cid, charge in charges.items():
            if charge.point_id == target:
                exclude_sources.add(cid)
        return target, exclude_sources

    raise ValueError(f"Unknown electric-field target: {target}")


def _net_electric_field_at(
    query: dict[str, Any],
    points: dict[str, Point],
    charges: dict[str, Charge],
    k_eff: float,
) -> Vec2:
    target_point_id, exclude_sources = _target_point_and_exclusions(query, points, charges)
    target_pos = points[target_point_id].position_m

    total = Vec2(0.0, 0.0)

    for source_id, source in charges.items():
        if source_id in exclude_sources:
            continue

        source_pos = points[source.point_id].position_m
        r_vec = target_pos - source_pos
        r = r_vec.norm()

        if r == 0:
            raise ValueError(
                f"Cannot compute field from charge {source_id} at its own point {target_point_id}."
            )

        # Electric field due to source charge:
        # E = k q_source (r_target - r_source) / |r|^3
        total = total + r_vec.scale(k_eff * source.charge_C / (r**3))

    return total


def _build_vectors(schema: dict[str, Any]) -> dict[str, VectorInput]:
    vectors: dict[str, VectorInput] = {}

    for i, item in enumerate(schema.get("vectors", []) or []):
        magnitude_data = item["magnitude"]
        mag_si = _quantity_to_si(magnitude_data)
        angle_deg = float(item.get("angle_deg", 0.0 if i == 0 else 0.0))
        theta = math.radians(angle_deg)
        vectors[item["id"]] = VectorInput(
            id=item["id"],
            value=Vec2(mag_si * math.cos(theta), mag_si * math.sin(theta)),
            unit=str(magnitude_data["unit"]),
        )

    return vectors


def _resultant_vector(vectors: dict[str, VectorInput]) -> Vec2:
    total = Vec2(0.0, 0.0)
    for item in vectors.values():
        total = total + item.value
    return total


def _format_value(value_si: float, si_unit: str, output_unit: Optional[str]) -> str:
    if output_unit is None:
        output_unit = si_unit
    value = convert(value_si, si_unit, output_unit)
    return f"{value:g} {output_unit}"


def _format_vector(vec: Vec2, si_unit: str, output_unit: Optional[str]):
    return {
        "Fx" if si_unit == "N" else "Ex": _format_value(vec.x, si_unit, output_unit),
        "Fy" if si_unit == "N" else "Ey": _format_value(vec.y, si_unit, output_unit),
        "magnitude": _format_value(vec.norm(), si_unit, output_unit),
    }


def _answer_vector(vec: Vec2, output: str, si_unit: str, unit: str):
    if output == "magnitude":
        return _format_value(vec.norm(), si_unit, unit)
    if output == "x_component":
        return _format_value(vec.x, si_unit, unit)
    if output == "y_component":
        return _format_value(vec.y, si_unit, unit)
    if output == "components":
        return _format_vector(vec, si_unit, unit)
    raise ValueError(f"Unsupported output: {output}")


def _answer_query(
    query: dict[str, Any],
    points: dict[str, Point],
    charges: dict[str, Charge],
    vectors: dict[str, VectorInput],
    k_eff: float,
):
    qtype = query.get("type")
    target = query.get("target")
    output = query.get("output", "magnitude")

    if qtype == "net_force":
        force = _net_force_on(str(target), points, charges, k_eff=k_eff)
        return _answer_vector(force, output, "N", query.get("unit", "N"))

    if qtype == "electric_field":
        field = _net_electric_field_at(query, points, charges, k_eff=k_eff)
        return _answer_vector(field, output, "V/m", query.get("unit", "V/m"))

    if qtype == "resultant_vector":
        result = _resultant_vector(vectors)
        return _answer_vector(result, output, "N", query.get("unit", "N"))

    raise ValueError(f"Unsupported query type: {qtype}")


def solve_schema(schema: dict[str, Any]) -> SolveResult:
    result = SolveResult(status="solved", domain="electrostatics")

    try:
        _repair_schema_before_validation(schema)
        validate_schema(schema)

        qtypes = [q.get("type") for q in schema.get("queries", [])]
        vector_only = bool(qtypes) and all(qtype == "resultant_vector" for qtype in qtypes)

        points: dict[str, Point] = {} if vector_only else _build_points(schema)
        charges = {} if vector_only else _build_charges(schema)
        vectors = _build_vectors(schema)
        k_eff = _k_eff(schema)

        result.add_step(
            "Build electrostatic system",
            "Built canonical vectors/points/charges from schema.",
            points={
                pid: {"x_m": point.position_m.x, "y_m": point.position_m.y}
                for pid, point in points.items()
            },
            charges={
                cid: {"charge_C": charge.charge_C, "at": charge.point_id}
                for cid, charge in charges.items()
            },
            vectors={
                vid: {"x": vec.value.x, "y": vec.value.y}
                for vid, vec in vectors.items()
            },
            relative_permittivity=_relative_permittivity(schema),
        )

        queries = schema.get("queries", [])
        if not queries:
            raise ValueError("Schema has no query.")

        answer = _answer_query(queries[0], points, charges, vectors, k_eff=k_eff)
        result.answer = answer

        result.add_step("Final answer", f"The requested {queries[0].get('type')} is {answer}.")
        return result

    except Exception as exc:
        result.status = "solve_failed"
        result.error = str(exc)
        return result
