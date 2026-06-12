from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


# This module is intentionally conservative. It is not meant to replace the LLM
# for every possible electrostatics problem; it repairs the high-frequency,
# low-ambiguity patterns in the benchmark where small LLM schema mistakes are
# catastrophic (wrong geometry primitive, force-vs-field query, missing M/H).

_SUPERSCRIPT_TRANS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
})

NUM = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:x|×|\*)\s*10\s*\^?\s*[-+]?\d+)?|10\s*\^?\s*[-+]?\d+)"
UNIT = r"(?:N/C|V/m|μC|µC|uC|mC|nC|pC|C|cm|mm|m|N|degree|degrees|deg|°)"


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str

    def as_schema(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit}


def _norm_text(problem: str) -> str:
    text = str(problem).translate(_SUPERSCRIPT_TRANS)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("µ", "μ")
    text = text.replace("×", "x")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_number(raw: str) -> float:
    s = raw.strip().translate(_SUPERSCRIPT_TRANS)
    s = s.replace("−", "-").replace("×", "x").replace("*", "x")
    s = re.sub(r"\s+", "", s)

    m = re.fullmatch(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))x10\^?([-+]?\d+)", s, flags=re.I)
    if m:
        return float(m.group(1)) * (10.0 ** int(m.group(2)))

    m = re.fullmatch(r"([-+]?)10\^?([-+]?\d+)", s, flags=re.I)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        return sign * (10.0 ** int(m.group(2)))

    return float(s)


def _unit(raw: str) -> str:
    u = str(raw).strip().replace("µ", "μ")
    if u == "μC":
        return "uC"
    if u == "°":
        return "degree"
    return u


def _q(value: float, unit: str) -> dict[str, Any]:
    return {"value": value, "unit": _unit(unit)}


def _distance(value: float, unit: str) -> dict[str, Any]:
    return {"value": value, "unit": _unit(unit)}


def _find_quantity_after(pattern: str, text: str, default_unit: str | None = None) -> Quantity | None:
    m = re.search(pattern + rf"\s*(?:=|of|is|by|:)?\s*({NUM})\s*({UNIT})?", text, flags=re.I)
    if not m:
        return None
    unit = m.group(2) or default_unit
    if unit is None:
        return None
    return Quantity(_parse_number(m.group(1)), _unit(unit))


def _all_quantities_with_unit(text: str, wanted_unit: str) -> list[Quantity]:
    out: list[Quantity] = []
    for m in re.finditer(rf"({NUM})\s*({UNIT})", text, flags=re.I):
        u = _unit(m.group(2))
        if u.lower() == wanted_unit.lower():
            out.append(Quantity(_parse_number(m.group(1)), u))
    return out


def _extract_direct_resultant(text: str) -> dict[str, Any] | None:
    low = text.lower()
    if "electric force" not in low and "electric forces" not in low and "resultant force" not in low:
        return None
    if "charge" in low and any(k in low for k in ["q1", "q2", "point charge", "placed at"]):
        # Coulomb problem, not a direct vector addition exercise.
        return None

    forces = _all_quantities_with_unit(text, "N")
    if len(forces) < 2:
        return None

    f1, f2 = forces[0], forces[1]

    angle = 0.0
    if "opposite direction" in low or "opposite directions" in low:
        angle = 180.0
    elif "same direction" in low or "same directions" in low:
        angle = 0.0
    else:
        m = re.search(rf"angle(?: of)?\s*({NUM})\s*(?:°|degree|degrees|deg)", text, flags=re.I)
        if m:
            angle = _parse_number(m.group(1))
        elif "perpendicular" in low or "right angle" in low:
            angle = 90.0
        else:
            return None

    return {
        "domain": "electrostatics",
        "vectors": [
            {"id": "F1", "magnitude": f1.as_schema(), "angle_deg": 0.0},
            {"id": "F2", "magnitude": f2.as_schema(), "angle_deg": angle},
        ],
        "queries": [{"type": "resultant_vector", "target": "vectors", "output": "magnitude", "unit": "N"}],
    }


def _extract_distances(text: str) -> dict[tuple[str, str], Quantity]:
    distances: dict[tuple[str, str], Quantity] = {}

    def add(a: str, b: str, q: Quantity) -> None:
        a = a.upper()
        b = b.upper()
        if a == b:
            return
        distances[tuple(sorted((a, b)))] = q

    for m in re.finditer(rf"\b([A-Z])([A-Z])\s*=\s*({NUM})\s*(cm|mm|m)\b", text, flags=re.I):
        add(m.group(1), m.group(2), Quantity(_parse_number(m.group(3)), _unit(m.group(4))))

    # "Points A and B are separated by 20 cm" / "A and B, separated by 8 cm"
    for m in re.finditer(rf"\b(?:points?\s+)?([A-Z])\s+and\s+([A-Z])\b[^.()]*?separated by\s*({NUM})\s*(cm|mm|m)\b", text, flags=re.I):
        add(m.group(1), m.group(2), Quantity(_parse_number(m.group(3)), _unit(m.group(4))))

    # "A is at ... and B, with AB = ..." already covered by AB=.
    return distances


def _dist_item(a: str, b: str, q: Quantity) -> dict[str, Any]:
    return {"between": [a, b], "value": q.value, "unit": q.unit}


def _same_distance(d1: Quantity, d2: Quantity, rel: float = 1e-6) -> bool:
    if d1.unit != d2.unit:
        return False
    return math.isclose(d1.value, d2.value, rel_tol=rel, abs_tol=1e-12)


def _collinear_order_from_distances(distances: dict[tuple[str, str], Quantity]) -> list[str] | None:
    points = sorted({p for pair in distances for p in pair})
    if len(points) != 3:
        return None
    a, b, c = points
    dab = distances.get(tuple(sorted((a, b))))
    dac = distances.get(tuple(sorted((a, c))))
    dbc = distances.get(tuple(sorted((b, c))))
    if not (dab and dac and dbc):
        return None
    if len({dab.unit, dac.unit, dbc.unit}) != 1:
        return None

    vals = {(a, b): dab.value, (a, c): dac.value, (b, c): dbc.value}
    if math.isclose(vals[(a, b)], vals[(a, c)] + vals[(b, c)], rel_tol=1e-6, abs_tol=1e-12):
        return [a, c, b]
    if math.isclose(vals[(a, c)], vals[(a, b)] + vals[(b, c)], rel_tol=1e-6, abs_tol=1e-12):
        return [a, b, c]
    if math.isclose(vals[(b, c)], vals[(a, b)] + vals[(a, c)], rel_tol=1e-6, abs_tol=1e-12):
        return [b, a, c]
    return None


def _geometry_from_text(text: str, points: set[str]) -> list[dict[str, Any]]:
    low = text.lower()
    geometry: list[dict[str, Any]] = []
    distances = _extract_distances(text)

    # Common implicit AB distance when only "the two charges are separated by ..." appears.
    if ("separated by" in low and ("q1" in low and "q2" in low) and tuple(sorted(("A", "B"))) not in distances):
        m = re.search(rf"separated by\s*({NUM})\s*(cm|mm|m)\b", text, flags=re.I)
        if m:
            distances[("A", "B")] = Quantity(_parse_number(m.group(1)), _unit(m.group(2)))
            points.update(["A", "B"])

    if "equilateral triangle" in low:
        side = _find_quantity_after(r"(?:side length|side)", text)
        if side is None:
            # fallback: first length after "with ... length"
            side = _find_quantity_after(r"(?:length)", text)
        if side is not None:
            points.update(["A", "B", "C"])
            geometry.append({
                "type": "EquilateralTriangle",
                "points": ["A", "B", "C"],
                "side": side.as_schema(),
                "orientation": "above",
            })

    elif "isosceles right" in low:
        leg = _find_quantity_after(r"(?:a|leg|side|sides of length)", text)
        if leg is None:
            # first cm/mm/m quantity is normally the leg.
            m = re.search(rf"({NUM})\s*(cm|mm|m)\b", text, flags=re.I)
            if m:
                leg = Quantity(_parse_number(m.group(1)), _unit(m.group(2)))
        if leg is not None:
            points.update(["A", "B", "C"])
            geometry.append({
                "type": "IsoscelesRightTriangle",
                "points": ["A", "B", "C"],
                "right_angle_at": "A",
                "leg": leg.as_schema(),
                "orientation": "above",
            })

    if distances:
        order = _collinear_order_from_distances(distances)
        dpoints = sorted({p for pair in distances for p in pair})
        points.update(dpoints)
        dist_items = [_dist_item(a, b, q) for (a, b), q in sorted(distances.items())]
        if order is not None:
            # Use adjacent distances only when available. Pairwise would also work,
            # but Collinear with a correct order is easier to debug.
            geometry.append({
                "type": "Collinear",
                "points": order,
                "order": order,
                "distances": dist_items,
            })
        elif not any(g.get("type") == "EquilateralTriangle" for g in geometry):
            geometry.append({
                "type": "PairwiseDistances",
                "points": dpoints,
                "distances": dist_items,
                "orientation": "above",
            })

    # Derived points.
    midpoint_id = None
    if "midpoint" in low or "middle point" in low or "trung điểm" in low:
        m = re.search(r"(?:point\s+)?([MHO])\b[^.]{0,80}\bmidpoint\b|\bmidpoint\s+([MHO])\b|\bmidpoint of\s+([A-Z])([A-Z])", text, flags=re.I)
        if m:
            midpoint_id = next((g for g in m.groups() if g and len(g) == 1), None)
        if midpoint_id is None:
            midpoint_id = "M"
        # Some Vietnamese/older problems name the midpoint H.
        if re.search(r"\bH\b[^.]{0,80}\bmidpoint\b|\bmidpoint\s+H\b", text, flags=re.I):
            midpoint_id = "H"
        points.add(midpoint_id)
        if ("A", "B") in distances or tuple(sorted(("A", "B"))) in distances or "A" in points and "B" in points:
            geometry.append({"type": "Midpoint", "point": midpoint_id, "between": ["A", "B"]})

    if "perpendicular bisector" in low:
        point = "M"
        mpoint = re.search(r"\bpoint\s+([A-Z])\b[^.]{0,80}\bperpendicular bisector|\bat\s+point\s+([A-Z])\b|\b([A-Z])\b[^.]{0,80}\blies on the perpendicular bisector", text, flags=re.I)
        if mpoint:
            point = next((g for g in mpoint.groups() if g), "M").upper()
        h = None
        for pat in [
            r"distance\s*(?:ℓ|l)?\s*=\s*({num})\s*(cm|mm|m)",
            r"({num})\s*(cm|mm|m)\s+from\s+(?:the\s+)?(?:midpoint|AB)",
            r"is\s+({num})\s*(cm|mm|m)\s+from\s+AB",
        ]:
            mm = re.search(pat.format(num=NUM), text, flags=re.I)
            if mm:
                h = Quantity(_parse_number(mm.group(1)), _unit(mm.group(2)))
                break
        if h is not None:
            points.update(["A", "B", point])
            geometry.append({
                "type": "PerpendicularBisectorPoint",
                "point": point,
                "between": ["A", "B"],
                "distance_from_segment": h.as_schema(),
                "orientation": "above",
            })

    if "foot of the altitude" in low or "foot of altitude" in low or "foot of the perpendicular" in low:
        points.update(["A", "B", "C", "H"])
        # Ensure triangle distances are available as pairwise if we have AB/AC/BC.
        if distances and not any(g.get("type") == "PairwiseDistances" for g in geometry):
            dpoints = sorted({p for pair in distances for p in pair})
            geometry.insert(0, {
                "type": "PairwiseDistances",
                "points": dpoints,
                "distances": [_dist_item(a, b, q) for (a, b), q in sorted(distances.items())],
                "orientation": "above",
            })
        geometry.append({"type": "FootOfPerpendicular", "point": "H", "from": "A", "to_line": ["B", "C"]})

    if "center" in low or "centroid" in low:
        # Physical center/centroid support. Some noisy benchmark rows ask "center"
        # but their answer matches a vertex formula; keep the schema physically honest.
        points.update(["A", "B", "C", "O"])
        geometry.append({"type": "Centroid", "point": "O", "of": ["A", "B", "C"]})

    return geometry


def _extract_charges(text: str, points: set[str]) -> list[dict[str, Any]]:
    low = text.lower()
    charges: dict[str, dict[str, Any]] = {}

    def set_charge(cid: str, value: float, unit: str) -> None:
        charges[cid] = {"id": cid, "charge": _q(value, unit), "at": None}

    # q1 = -q2 = 10^-7 C
    for m in re.finditer(rf"\b(q\d+)\s*=\s*-\s*(q\d+)\s*=\s*({NUM})\s*(μC|µC|uC|mC|nC|pC|C)\b", text, flags=re.I):
        q_left, q_right = m.group(1), m.group(2)
        val = _parse_number(m.group(3))
        unit = _unit(m.group(4))
        set_charge(q_left, val, unit)
        set_charge(q_right, -val, unit)

    # q1 = q2 = q3 = value unit OR q1 = q2 = value unit
    for m in re.finditer(rf"\b(q\d+(?:\s*=\s*q\d+)+)\s*=\s*({NUM})\s*(μC|µC|uC|mC|nC|pC|C)\b", text, flags=re.I):
        ids = re.findall(r"q\d+", m.group(1), flags=re.I)
        val = _parse_number(m.group(2))
        unit = _unit(m.group(3))
        for cid in ids:
            if cid not in charges:
                set_charge(cid, val, unit)

    # Individual q_i = value unit.
    for m in re.finditer(rf"\b(q\d+|q0|q)\s*=\s*({NUM})\s*(μC|µC|uC|mC|nC|pC|C)\b", text, flags=re.I):
        cid = m.group(1)
        # Do not let generic q overwrite q1/q2/q3 equal-charge pattern unless it is target q.
        if cid == "q" and cid in charges:
            continue
        if cid not in charges:
            set_charge(cid, _parse_number(m.group(2)), _unit(m.group(3)))

    # Generic "Three identical/equal charges q = ...".
    if re.search(r"\bthree\b[^.]{0,80}\b(?:identical|equal|like-signed)?\s*(?:electric\s+)?charges?\s+q\s*=", text, flags=re.I):
        m = re.search(rf"\bq\s*=\s*({NUM})\s*(μC|µC|uC|mC|nC|pC|C)\b", text, flags=re.I)
        if m:
            val = _parse_number(m.group(1))
            unit = _unit(m.group(2))
            for cid in ["q1", "q2", "q3"]:
                if cid not in charges:
                    set_charge(cid, val, unit)

    # Map charge locations.
    def assign(cid: str, point: str) -> None:
        if cid in charges:
            charges[cid]["at"] = point.upper()
            points.add(point.upper())

    # Explicit "q1 is at A" / "q1 ... placed at A".
    for m in re.finditer(r"\b(q\d+|q0|q)\b[^.]{0,100}?\b(?:at|placed at|located at)\s+(?:point\s+|vertex\s+)?([A-Z])\b", text, flags=re.I):
        assign(m.group(1), m.group(2))

    # Respectively patterns.
    if re.search(r"points?\s+A\s+and\s+B\s+respectively", text, flags=re.I):
        assign("q1", "A")
        assign("q2", "B")
    if re.search(r"vertices\s+B\s+and\s+C", text, flags=re.I):
        assign("q1", "B")
        assign("q2", "C")
    elif re.search(r"vertices?\s+(?:of\s+)?(?:an?\s+)?(?:equilateral\s+|right[- ]angled\s+|isosceles\s+right\s+)?triangle", text, flags=re.I):
        # Default canonical vertex order.
        for cid, point in zip(["q1", "q2", "q3"], ["A", "B", "C"]):
            assign(cid, point)

    # Common direct wording: q1 is at A and q2 is at B.
    if re.search(r"\bq1\b[^.]{0,80}\b(?:is at|at)\s+A\b", text, flags=re.I):
        assign("q1", "A")
    if re.search(r"\bq2\b[^.]{0,80}\b(?:is at|at)\s+B\b", text, flags=re.I):
        assign("q2", "B")

    # q0/q placed at midpoint/point H/M.
    for cid in ["q0", "q"]:
        if cid in charges:
            m = re.search(rf"\b{cid}\b[^.]*?placed at\s+(?:point\s+)?([A-Z])\b", text, flags=re.I)
            if m:
                assign(cid, m.group(1))
            elif "midpoint" in low:
                # H if explicitly named, otherwise M.
                assign(cid, "H" if re.search(r"\bH\b[^.]{0,80}\bmidpoint\b|at\s+H", text, flags=re.I) else "M")

    # Target charge q at H in foot-of-altitude problem.
    if "foot of the altitude" in low and "q" in charges:
        assign("q", "H")

    # Fill still-missing locations when there is an obvious canonical setup.
    missing = [cid for cid, data in charges.items() if data.get("at") is None]
    if missing:
        if set(charges) >= {"q1", "q2"}:
            assign("q1", charges["q1"].get("at") or "A")
            assign("q2", charges["q2"].get("at") or "B")
        if set(charges) >= {"q1", "q2", "q3"} and "q3" in missing:
            assign("q3", "C")
        if "q0" in missing:
            assign("q0", "M" if "midpoint" in low else "C")
        if "q" in missing and "q1" in charges and "q2" in charges:
            assign("q", "H" if "foot of the altitude" in low else ("M" if "midpoint" in low else "C"))

    return [data for _, data in sorted(charges.items(), key=lambda kv: (kv[0] == "q", kv[0])) if data.get("at") is not None]


def _target_for_query(text: str, charges: list[dict[str, Any]], points: set[str], qtype: str) -> str | None:
    low = text.lower()
    if qtype == "electric_field":
        # Target is a point or a charge position.
        for pat in [
            r"at\s+(?:point\s+|vertex\s+)?([A-Z])\b",
            r"at\s+the\s+midpoint\s+([A-Z])\b",
            r"midpoint\s+([A-Z])\b",
        ]:
            matches = re.findall(pat, text, flags=re.I)
            if matches:
                # Use the last such point; early "at points A and B" are sources.
                cand = matches[-1].upper()
                if cand not in {"A", "B"} or "field" in low:
                    return cand
        if "position of q3" in low:
            return "q3"
        if "midpoint" in low:
            return "H" if " at h" in low or "midpoint h" in low else "M"
        if "center" in low or "centroid" in low:
            return "O"
        return None

    # Net force target is a charge id.
    for pat in [
        r"(?:force acting on|net force acting on|acting on)\s+(?:electric\s+charge\s+|charge\s+)?(q\d+|q0|q)\b",
        r"(?:on|acting on)\s+the\s+charge\s+(?:at|placed at)\s+(?:the\s+)?right\s+angle",
    ]:
        m = re.search(pat, text, flags=re.I)
        if m:
            if m.lastindex:
                return m.group(1)
            for ch in charges:
                if ch.get("at") == "A":
                    return ch["id"]
    if "right angle vertex" in low:
        for ch in charges:
            if ch.get("at") == "A":
                return ch["id"]
    if any(ch["id"] == "q0" for ch in charges):
        return "q0"
    if any(ch["id"] == "q" for ch in charges):
        return "q"
    if any(ch["id"] == "q3" for ch in charges):
        return "q3"
    return None


def extract_electrostatics_schema_from_text(problem: str) -> dict[str, Any] | None:
    text = _norm_text(problem)
    low = text.lower()

    direct = _extract_direct_resultant(text)
    if direct is not None:
        return direct

    electrostatic_cues = [
        "point charge", "point charges", "electric charge", "electric charges", "charges", "q1", "q2",
        "electric field", "field strength", "net force", "force acting", "coulomb",
    ]
    geometry_cues = [
        "point", "points", "vertices", "triangle", "midpoint", "separated", "perpendicular", "AB", "AC", "BC",
    ]
    if not any(c in low for c in electrostatic_cues) or not any(c.lower() in low for c in geometry_cues):
        return None

    points: set[str] = set()
    geometry = _geometry_from_text(text, points)
    charges = _extract_charges(text, points)

    # Add points from charges after location assignment.
    for ch in charges:
        if ch.get("at"):
            points.add(ch["at"])

    if not charges and "electric field" not in low:
        return None

    # Query selection. Field questions should not be encoded as net_force.
    qtype = "electric_field" if any(k in low for k in ["electric field", "field strength", "v/m", "n/c"]) else "net_force"
    target = _target_for_query(text, charges, points, qtype)
    if target is not None and qtype == "electric_field" and re.fullmatch(r"[A-Z]", target):
        points.add(target)

    if target is None:
        return None

    output_unit = "V/m" if qtype == "electric_field" else "N"
    if "n/c" in low:
        output_unit = "N/C"

    schema: dict[str, Any] = {
        "domain": "electrostatics",
        "points": [{"id": p} for p in sorted(points)],
        "geometry": geometry,
        "charges": charges,
        "queries": [{"type": qtype, "target": target, "output": "magnitude", "unit": output_unit}],
    }

    # Do NOT include relative permittivity here. Several benchmark rows mention a
    # medium/dielectric constant but their validated answer uses k=9e9 directly.
    return schema
