from __future__ import annotations

import math
from typing import Any

from xai_physics.core.units import to_si
from xai_physics.domains.electrostatics.vector import Vec2


_EPS = 1e-10


def _quantity_to_si(data: dict[str, Any]) -> float:
    return to_si(float(data["value"]), str(data["unit"]))


def _derived_point_ids(schema: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for geom in schema.get("geometry", []) or []:
        if geom.get("type") in {"Midpoint", "PerpendicularBisectorPoint", "Centroid", "FootOfPerpendicular"}:
            point = geom.get("point")
            if isinstance(point, str) and point not in ids:
                ids.append(point)
    return ids


def _repair_missing_point_aliases(schema: dict[str, Any], coords: dict[str, Vec2]) -> None:
    """Copy coordinates from obvious LLM aliases like m/h to requested M.

    Qwen sometimes emits both point M and point m/h, builds geometry for m/h,
    but asks the query at M. Do not fail geometry for that; map M to the only
    derived coordinate when it is unambiguous.
    """
    point_ids = {p.get("id") for p in schema.get("points", []) if isinstance(p, dict)}
    missing = sorted(pid for pid in point_ids if isinstance(pid, str) and pid not in coords)
    if not missing:
        return

    derived = [pid for pid in _derived_point_ids(schema) if pid in coords]

    for pid in missing:
        candidates: list[str] = []
        low = pid.lower()
        if low in coords and low != pid:
            candidates.append(low)

        if not candidates:
            # Prefer one-letter lowercase placeholders produced by the LLM.
            lower_derived = [d for d in derived if d != pid and len(d) == 1 and d.islower()]
            if len(lower_derived) == 1:
                candidates.append(lower_derived[0])

        if not candidates and len(derived) == 1 and derived[0] != pid:
            candidates.append(derived[0])

        if candidates:
            coords[pid] = coords[candidates[0]]


def _explicit_points(schema: dict[str, Any]) -> dict[str, Vec2]:
    coords: dict[str, Vec2] = {}

    for point in schema.get("points", []):
        if "x" in point and "y" in point:
            coords[point["id"]] = Vec2(
                x=_quantity_to_si(point["x"]),
                y=_quantity_to_si(point["y"]),
            )

    return coords


def _add_distance(
    dist_map: dict[tuple[str, str], float],
    a: str,
    b: str,
    d: float,
) -> None:
    dist_map[(a, b)] = d
    dist_map[(b, a)] = d


def _collect_distances(schema: dict[str, Any]) -> dict[tuple[str, str], float]:
    dist_map: dict[tuple[str, str], float] = {}

    for geom in schema.get("geometry", []) or []:
        gtype = geom.get("type")

        if gtype in {"PairwiseDistances", "Collinear"}:
            for item in geom.get("distances", []) or []:
                pair = item.get("between")
                if isinstance(pair, list) and len(pair) == 2:
                    _add_distance(
                        dist_map,
                        str(pair[0]),
                        str(pair[1]),
                        _quantity_to_si({"value": item["value"], "unit": item["unit"]}),
                    )

        elif gtype == "EquilateralTriangle":
            points = geom.get("points", [])
            if len(points) == 3 and "side" in geom:
                side = _quantity_to_si(geom["side"])
                a, b, c = points
                _add_distance(dist_map, a, b, side)
                _add_distance(dist_map, a, c, side)
                _add_distance(dist_map, b, c, side)

        elif gtype == "IsoscelesRightTriangle":
            points = geom.get("points", [])
            right = geom.get("right_angle_at")
            if len(points) == 3 and right in points and "leg" in geom:
                leg = _quantity_to_si(geom["leg"])
                others = [p for p in points if p != right]
                _add_distance(dist_map, right, others[0], leg)
                _add_distance(dist_map, right, others[1], leg)
                _add_distance(dist_map, others[0], others[1], leg * math.sqrt(2.0))

    return dist_map


def _ensure_segment(
    coords: dict[str, Vec2],
    dist_map: dict[tuple[str, str], float],
    a_id: str,
    b_id: str,
) -> bool:
    """Ensure two anchor points exist. Returns True when it created something."""
    if a_id in coords and b_id in coords:
        return False

    d = dist_map.get((a_id, b_id))
    if d is None:
        return False

    changed = False

    if a_id not in coords and b_id not in coords:
        coords[a_id] = Vec2(0.0, 0.0)
        coords[b_id] = Vec2(d, 0.0)
        return True

    if a_id in coords and b_id not in coords:
        coords[b_id] = Vec2(coords[a_id].x + d, coords[a_id].y)
        changed = True

    if b_id in coords and a_id not in coords:
        coords[a_id] = Vec2(coords[b_id].x - d, coords[b_id].y)
        changed = True

    return changed


def _place_third_by_distances(
    coords: dict[str, Vec2],
    dist_map: dict[tuple[str, str], float],
    a_id: str,
    b_id: str,
    c_id: str,
    orientation: str = "above",
) -> bool:
    if c_id in coords:
        return False

    if a_id not in coords or b_id not in coords:
        if not _ensure_segment(coords, dist_map, a_id, b_id):
            return False

    d_ab = (coords[b_id] - coords[a_id]).norm()
    d_ac = dist_map.get((a_id, c_id))
    d_bc = dist_map.get((b_id, c_id))

    if d_ab <= 0 or d_ac is None or d_bc is None:
        return False

    a = coords[a_id]
    b = coords[b_id]
    ux = (b.x - a.x) / d_ab
    uy = (b.y - a.y) / d_ab

    # Unit perpendicular to AB.
    px = -uy
    py = ux

    x_along = (d_ac * d_ac + d_ab * d_ab - d_bc * d_bc) / (2.0 * d_ab)
    h2 = d_ac * d_ac - x_along * x_along

    # Degenerate valid triangle => collinear. Slight negative is usually rounding.
    if h2 < 0 and abs(h2) <= max(1.0, d_ac * d_ac) * 1e-8:
        h2 = 0.0
    if h2 < 0:
        raise ValueError(
            f"Inconsistent pairwise distances for {a_id}, {b_id}, {c_id}."
        )

    sign = -1.0 if orientation == "below" else 1.0
    height = sign * math.sqrt(h2)

    coords[c_id] = Vec2(
        a.x + ux * x_along + px * height,
        a.y + uy * x_along + py * height,
    )
    return True


def _build_equilateral(
    geom: dict[str, Any],
    coords: dict[str, Vec2],
    dist_map: dict[tuple[str, str], float],
) -> bool:
    points = geom["points"]
    if len(points) != 3:
        raise ValueError("EquilateralTriangle requires exactly three points.")

    a_id, b_id, c_id = points
    side = _quantity_to_si(geom["side"])
    orientation = geom.get("orientation", "above")

    _add_distance(dist_map, a_id, b_id, side)
    _add_distance(dist_map, a_id, c_id, side)
    _add_distance(dist_map, b_id, c_id, side)

    changed = _ensure_segment(coords, dist_map, a_id, b_id)
    changed = _place_third_by_distances(
        coords, dist_map, a_id, b_id, c_id, orientation=orientation
    ) or changed
    return changed


def _build_isosceles_right(
    geom: dict[str, Any],
    coords: dict[str, Vec2],
    dist_map: dict[tuple[str, str], float],
) -> bool:
    points = geom["points"]
    if len(points) != 3:
        raise ValueError("IsoscelesRightTriangle requires exactly three points.")

    right = geom.get("right_angle_at")
    if right not in points:
        raise ValueError("IsoscelesRightTriangle.right_angle_at must be one of points.")

    leg = _quantity_to_si(geom["leg"])
    orientation = geom.get("orientation", "above")
    others = [p for p in points if p != right]
    b_id, c_id = others[0], others[1]

    if right in coords and b_id in coords and c_id in coords:
        return False

    if right not in coords:
        coords[right] = Vec2(0.0, 0.0)

    changed = False
    if b_id not in coords:
        coords[b_id] = Vec2(coords[right].x + leg, coords[right].y)
        changed = True

    sign = -1.0 if orientation == "below" else 1.0
    if c_id not in coords:
        coords[c_id] = Vec2(coords[right].x, coords[right].y + sign * leg)
        changed = True

    _add_distance(dist_map, right, b_id, leg)
    _add_distance(dist_map, right, c_id, leg)
    _add_distance(dist_map, b_id, c_id, leg * math.sqrt(2.0))
    return changed


def _build_pairwise(
    geom: dict[str, Any],
    coords: dict[str, Vec2],
    dist_map: dict[tuple[str, str], float],
) -> bool:
    points = list(geom.get("points", []))
    orientation = geom.get("orientation", "above")
    changed = False

    for item in geom.get("distances", []) or []:
        pair = item["between"]
        _add_distance(
            dist_map,
            pair[0],
            pair[1],
            _quantity_to_si({"value": item["value"], "unit": item["unit"]}),
        )

    if len(points) == 2:
        return _ensure_segment(coords, dist_map, points[0], points[1])

    if len(points) >= 3:
        # Try every pair as anchor and every other point as the third point.
        for a_id in points:
            for b_id in points:
                if a_id == b_id or (a_id, b_id) not in dist_map:
                    continue
                if _ensure_segment(coords, dist_map, a_id, b_id):
                    changed = True
                for c_id in points:
                    if c_id in {a_id, b_id}:
                        continue
                    if (a_id, c_id) in dist_map and (b_id, c_id) in dist_map:
                        if _place_third_by_distances(
                            coords, dist_map, a_id, b_id, c_id, orientation=orientation
                        ):
                            changed = True
        return changed

    return False


def _build_collinear(
    geom: dict[str, Any],
    coords: dict[str, Vec2],
    dist_map: dict[tuple[str, str], float],
) -> bool:
    order = geom.get("order") or geom.get("points")
    if not isinstance(order, list) or len(order) < 2:
        raise ValueError("Collinear geometry requires order or points with at least two ids.")

    for item in geom.get("distances", []) or []:
        pair = item["between"]
        _add_distance(
            dist_map,
            pair[0],
            pair[1],
            _quantity_to_si({"value": item["value"], "unit": item["unit"]}),
        )

    first = order[0]
    if first not in coords:
        coords[first] = Vec2(0.0, 0.0)

    changed = False
    for left, right in zip(order, order[1:]):
        key = (left, right)
        if key not in dist_map:
            raise ValueError(f"Missing distance between adjacent points {left}-{right}.")
        if right not in coords:
            coords[right] = Vec2(coords[left].x + dist_map[key], coords[first].y)
            changed = True

    return changed


def _build_midpoint(
    geom: dict[str, Any],
    coords: dict[str, Vec2],
    dist_map: dict[tuple[str, str], float],
) -> bool:
    point_id = geom.get("point")
    between = geom.get("between", [])
    if not isinstance(point_id, str) or len(between) != 2:
        raise ValueError("Midpoint requires point and between=[A,B].")

    a_id, b_id = between
    _ensure_segment(coords, dist_map, a_id, b_id)

    if a_id not in coords or b_id not in coords:
        raise ValueError(f"Cannot build midpoint {point_id}; missing endpoints {a_id}, {b_id}.")

    old = coords.get(point_id)
    coords[point_id] = Vec2(
        (coords[a_id].x + coords[b_id].x) / 2.0,
        (coords[a_id].y + coords[b_id].y) / 2.0,
    )
    return old != coords[point_id]


def _build_point_on_line(
    geom: dict[str, Any],
    coords: dict[str, Vec2],
    dist_map: dict[tuple[str, str], float],
) -> bool:
    point_id = geom.get("point")
    start = geom.get("start")
    end = geom.get("end")
    if not all(isinstance(x, str) for x in [point_id, start, end]):
        raise ValueError("PointOnLine requires point, start, and end.")

    _ensure_segment(coords, dist_map, start, end)
    if start not in coords or end not in coords:
        raise ValueError(f"Cannot build PointOnLine {point_id}; missing {start}-{end}.")

    d = _quantity_to_si(geom["distance_from_start"])
    direction = geom.get("direction", "toward_end")

    start_pos = coords[start]
    end_pos = coords[end]
    axis = end_pos - start_pos
    axis_norm = axis.norm()
    if axis_norm == 0:
        raise ValueError("PointOnLine endpoints coincide.")

    unit = axis.scale(1.0 / axis_norm)
    sign = -1.0 if direction == "away_from_end" else 1.0
    old = coords.get(point_id)
    coords[point_id] = start_pos + unit.scale(sign * d)
    return old != coords[point_id]


def _build_perpendicular_bisector_point(
    geom: dict[str, Any],
    coords: dict[str, Vec2],
    dist_map: dict[tuple[str, str], float],
) -> bool:
    point_id = geom.get("point")
    between = geom.get("between", [])
    if not isinstance(point_id, str) or len(between) != 2:
        raise ValueError("PerpendicularBisectorPoint requires point and between=[A,B].")

    a_id, b_id = between
    _ensure_segment(coords, dist_map, a_id, b_id)
    if a_id not in coords or b_id not in coords:
        raise ValueError(
            f"Cannot build perpendicular-bisector point {point_id}; missing endpoints."
        )

    h = _quantity_to_si(geom["distance_from_segment"])
    orientation = geom.get("orientation", "above")
    sign = -1.0 if orientation == "below" else 1.0

    a = coords[a_id]
    b = coords[b_id]
    axis = b - a
    d = axis.norm()
    if d == 0:
        raise ValueError("PerpendicularBisectorPoint endpoints coincide.")

    mid = Vec2((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
    px = -axis.y / d
    py = axis.x / d

    old = coords.get(point_id)
    coords[point_id] = Vec2(mid.x + sign * px * h, mid.y + sign * py * h)
    return old != coords[point_id]


def _build_centroid(
    geom: dict[str, Any],
    coords: dict[str, Vec2],
    _dist_map: dict[tuple[str, str], float],
) -> bool:
    point_id = geom.get("point")
    of_points = geom.get("of", [])
    if not isinstance(point_id, str) or not isinstance(of_points, list) or len(of_points) < 2:
        raise ValueError("Centroid requires point and of=[...].")

    if not all(p in coords for p in of_points):
        return False

    old = coords.get(point_id)
    coords[point_id] = Vec2(
        sum(coords[p].x for p in of_points) / len(of_points),
        sum(coords[p].y for p in of_points) / len(of_points),
    )
    return old != coords[point_id]


def _build_foot_of_perpendicular(
    geom: dict[str, Any],
    coords: dict[str, Vec2],
    _dist_map: dict[tuple[str, str], float],
) -> bool:
    point_id = geom.get("point")
    from_id = geom.get("from")
    to_line = geom.get("to_line", [])
    if not isinstance(point_id, str) or not isinstance(from_id, str) or len(to_line) != 2:
        raise ValueError("FootOfPerpendicular requires point, from, and to_line=[B,C].")

    b_id, c_id = to_line
    if from_id not in coords or b_id not in coords or c_id not in coords:
        return False

    a = coords[from_id]
    b = coords[b_id]
    c = coords[c_id]
    bc = c - b
    denom = bc.x * bc.x + bc.y * bc.y
    if denom == 0:
        raise ValueError("FootOfPerpendicular line endpoints coincide.")

    t = ((a.x - b.x) * bc.x + (a.y - b.y) * bc.y) / denom
    old = coords.get(point_id)
    coords[point_id] = Vec2(b.x + t * bc.x, b.y + t * bc.y)
    return old != coords[point_id]


def _build_perpendicular_rays_from_point(
    geom: dict[str, Any],
    coords: dict[str, Vec2],
    _dist_map: dict[tuple[str, str], float],
) -> bool:
    """Place source points on perpendicular rays from a center point.

    Useful for problems saying two fields at M are perpendicular and sources are each
    a known distance from M. Canonical placement: first point on +x, second on +y.
    """
    center = geom.get("center")
    points = geom.get("points", [])
    distances = geom.get("distances", [])
    if not isinstance(center, str) or len(points) != len(distances):
        raise ValueError("PerpendicularRaysFromPoint requires center, points, distances.")

    if center not in coords:
        coords[center] = Vec2(0.0, 0.0)

    directions = [Vec2(1.0, 0.0), Vec2(0.0, 1.0), Vec2(-1.0, 0.0), Vec2(0.0, -1.0)]
    changed = False
    for i, point_id in enumerate(points):
        if point_id in coords:
            continue
        d = _quantity_to_si(distances[i])
        u = directions[i % len(directions)]
        coords[point_id] = Vec2(coords[center].x + u.x * d, coords[center].y + u.y * d)
        changed = True
    return changed


def build_coordinates(schema: dict[str, Any]) -> dict[str, Vec2]:
    """
    Build canonical 2D coordinates from explicit coordinates and geometry relations.

    Supported geometry:
    - explicit points with x,y
    - EquilateralTriangle
    - IsoscelesRightTriangle
    - PairwiseDistances for 2 or 3 points (including degenerate collinear triples)
    - Collinear
    - Midpoint
    - PointOnLine
    - PerpendicularBisectorPoint
    - Centroid
    - FootOfPerpendicular
    - PerpendicularRaysFromPoint
    """
    coords = _explicit_points(schema)
    dist_map = _collect_distances(schema)
    geometry = schema.get("geometry", []) or []

    builders = {
        "EquilateralTriangle": _build_equilateral,
        "IsoscelesRightTriangle": _build_isosceles_right,
        "PairwiseDistances": _build_pairwise,
        "Collinear": _build_collinear,
        "Midpoint": _build_midpoint,
        "PointOnLine": _build_point_on_line,
        "PerpendicularBisectorPoint": _build_perpendicular_bisector_point,
        "Centroid": _build_centroid,
        "FootOfPerpendicular": _build_foot_of_perpendicular,
        "PerpendicularRaysFromPoint": _build_perpendicular_rays_from_point,
    }

    # Multiple passes allow derived points (midpoint/foot/centroid) to depend on
    # coordinates built by earlier or later geometry blocks.
    for _ in range(max(1, len(geometry) + 2)):
        changed = False
        for geom in geometry:
            gtype = geom.get("type")
            builder = builders.get(gtype)
            if builder is None:
                raise ValueError(f"Unsupported geometry type: {gtype}")
            changed = builder(geom, coords, dist_map) or changed
        if not changed:
            break

    _repair_missing_point_aliases(schema, coords)

    point_ids = {p["id"] for p in schema.get("points", [])}
    missing = sorted(point_ids - set(coords))
    if missing:
        raise ValueError(f"Cannot build coordinates for points: {missing}")

    return coords
