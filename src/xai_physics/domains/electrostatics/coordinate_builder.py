from __future__ import annotations

import math
from typing import Any

from xai_physics.core.units import to_si
from xai_physics.domains.electrostatics.vector import Vec2


def _quantity_to_si(data: dict[str, Any]) -> float:
    return to_si(float(data["value"]), str(data["unit"]))


def _explicit_points(schema: dict[str, Any]) -> dict[str, Vec2]:
    coords: dict[str, Vec2] = {}

    for point in schema.get("points", []):
        if "x" in point and "y" in point:
            coords[point["id"]] = Vec2(
                x=_quantity_to_si(point["x"]),
                y=_quantity_to_si(point["y"]),
            )

    return coords


def _build_equilateral(geom: dict[str, Any], coords: dict[str, Vec2]) -> None:
    points = geom["points"]
    if len(points) != 3:
        raise ValueError("EquilateralTriangle requires exactly three points.")

    a_id, b_id, c_id = points
    side = _quantity_to_si(geom["side"])
    orientation = geom.get("orientation", "above")

    # Canonical placement:
    # A = (0, 0), B = (side, 0), C = (side/2, ±sqrt(3)/2 side)
    if a_id not in coords:
        coords[a_id] = Vec2(0.0, 0.0)

    if b_id not in coords:
        coords[b_id] = Vec2(coords[a_id].x + side, coords[a_id].y)

    sign = -1.0 if orientation == "below" else 1.0

    if c_id not in coords:
        coords[c_id] = Vec2(
            coords[a_id].x + side / 2.0,
            coords[a_id].y + sign * math.sqrt(3.0) * side / 2.0,
        )


def _build_collinear(geom: dict[str, Any], coords: dict[str, Vec2]) -> None:
    order = geom.get("order") or geom.get("points")
    if not isinstance(order, list) or len(order) < 2:
        raise ValueError("Collinear geometry requires order or points with at least two ids.")

    distances = geom.get("distances", [])
    if not isinstance(distances, list):
        raise ValueError("Collinear.distances must be a list.")

    # Canonical placement on x-axis.
    first = order[0]
    if first not in coords:
        coords[first] = Vec2(0.0, 0.0)

    # Build a quick map for distance between adjacent ordered points.
    dist_map: dict[tuple[str, str], float] = {}
    for item in distances:
        pair = item["between"]
        if len(pair) != 2:
            raise ValueError("Distance.between must contain exactly two point ids.")
        d = _quantity_to_si({"value": item["value"], "unit": item["unit"]})
        dist_map[(pair[0], pair[1])] = d
        dist_map[(pair[1], pair[0])] = d

    x = coords[first].x

    for left, right in zip(order, order[1:]):
        key = (left, right)
        if key not in dist_map:
            raise ValueError(f"Missing distance between adjacent points {left}-{right}.")

        x = coords[left].x + dist_map[key]
        coords[right] = Vec2(x, coords[first].y)


def build_coordinates(schema: dict[str, Any]) -> dict[str, Vec2]:
    """
    Build canonical 2D coordinates from explicit point coordinates
    and simple geometry relations.

    Supported v0:
    - explicit points with x,y
    - EquilateralTriangle
    - Collinear
    """
    coords = _explicit_points(schema)

    for geom in schema.get("geometry", []):
        gtype = geom.get("type")

        if gtype == "EquilateralTriangle":
            _build_equilateral(geom, coords)
        elif gtype == "Collinear":
            _build_collinear(geom, coords)
        else:
            raise ValueError(f"Unsupported geometry type: {gtype}")

    point_ids = {p["id"] for p in schema.get("points", [])}

    missing = sorted(point_ids - set(coords))
    if missing:
        raise ValueError(f"Cannot build coordinates for points: {missing}")

    return coords
