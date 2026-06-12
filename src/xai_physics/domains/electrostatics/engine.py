from __future__ import annotations

from dataclasses import dataclass
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


def _quantity_to_si(data: dict[str, Any]) -> float:
    return to_si(float(data["value"]), str(data["unit"]))


def _build_points(schema: dict[str, Any]) -> dict[str, Point]:
    coords = build_coordinates(schema)
    return {
        point_id: Point(id=point_id, position_m=pos)
        for point_id, pos in coords.items()
    }


def _build_charges(schema: dict[str, Any]) -> dict[str, Charge]:
    charges: dict[str, Charge] = {}

    for item in schema.get("charges", []):
        charge = Charge(
            id=item["id"],
            charge_C=_quantity_to_si(item["charge"]),
            point_id=item["at"],
        )
        charges[charge.id] = charge

    return charges


def _net_force_on(
    target_id: str,
    points: dict[str, Point],
    charges: dict[str, Charge],
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
            raise ValueError(
                f"Charges {target_id} and {source_id} occupy the same point."
            )

        # Coulomb vector force on target due to source:
        # F = k q_target q_source (r_target - r_source) / |r|^3
        scale = K_COULOMB * target.charge_C * source.charge_C / (r ** 3)
        contribution = r_vec.scale(scale)
        total = total + contribution

    return total


def _format_value(value_si: float, si_unit: str, output_unit: Optional[str]) -> str:
    if output_unit is None:
        output_unit = si_unit
    value = convert(value_si, si_unit, output_unit)
    return f"{value:g} {output_unit}"


def _answer_query(
    query: dict[str, Any],
    points: dict[str, Point],
    charges: dict[str, Charge],
):
    qtype = query.get("type")
    target = query.get("target")
    output = query.get("output", "magnitude")
    unit = query.get("unit", "N")

    if qtype != "net_force":
        raise ValueError(f"Unsupported query type: {qtype}")

    force = _net_force_on(target, points, charges)

    if output == "magnitude":
        return _format_value(force.norm(), "N", unit)

    if output == "x_component":
        return _format_value(force.x, "N", unit)

    if output == "y_component":
        return _format_value(force.y, "N", unit)

    if output == "components":
        return {
            "Fx": _format_value(force.x, "N", unit),
            "Fy": _format_value(force.y, "N", unit),
            "magnitude": _format_value(force.norm(), "N", unit),
        }

    raise ValueError(f"Unsupported output: {output}")


def solve_schema(schema: dict[str, Any]) -> SolveResult:
    result = SolveResult(status="solved", domain="electrostatics")

    try:
        validate_schema(schema)

        points = _build_points(schema)
        charges = _build_charges(schema)

        result.add_step(
            "Build electrostatic system",
            "Built points and charges from canonical schema.",
            points={
                pid: {
                    "x_m": point.position_m.x,
                    "y_m": point.position_m.y,
                }
                for pid, point in points.items()
            },
            charges={
                cid: {
                    "charge_C": charge.charge_C,
                    "at": charge.point_id,
                }
                for cid, charge in charges.items()
            },
        )

        queries = schema.get("queries", [])
        if not queries:
            raise ValueError("Schema has no query.")

        answer = _answer_query(queries[0], points, charges)
        result.answer = answer

        result.add_step(
            "Final answer",
            f"The requested {queries[0].get('type')} is {answer}.",
        )

        return result

    except Exception as exc:
        result.status = "solve_failed"
        result.error = str(exc)
        return result
