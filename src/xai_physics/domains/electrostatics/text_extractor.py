from __future__ import annotations

import re
from typing import Any

NUM = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:x|×|\*)\s*10\s*\^?\s*[-+]?\d+)?|10\s*\^?\s*[-+]?\d+)"
UNIT_CHARGE = r"(?:μC|µC|uC|mC|nC|pC|C)"
UNIT_LEN = r"(?:cm|mm|m)"

_SUPERSCRIPT_TRANS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
})


def _norm(text: str) -> str:
    return (
        str(text)
        .translate(_SUPERSCRIPT_TRANS)
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("，", ",")
        .replace("µ", "μ")
    )


def _unit(raw: str) -> str:
    u = str(raw).strip().replace("µ", "μ")
    return u.replace("μC", "uC")


def _parse_number(raw: str) -> float:
    s = _norm(str(raw)).strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("×", "x")
    m = re.fullmatch(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)?)?(?:x|\*)?10\^?([-+]?\d+)", s, flags=re.I)
    if m:
        base_s = m.group(1)
        base = float(base_s) if base_s not in {None, "", "+", "-"} else ( -1.0 if base_s == "-" else 1.0 )
        return base * (10.0 ** int(m.group(2)))
    return float(s)


def _find(pattern: str, text: str) -> re.Match[str] | None:
    return re.search(pattern, text, flags=re.I)


def _quantity(value: float, unit: str) -> dict[str, Any]:
    return {"value": value, "unit": _unit(unit)}


def _length_quantity(value: float, unit: str) -> dict[str, Any]:
    return {"value": value, "unit": unit}


def _find_ab_distance(text: str) -> tuple[float, str] | None:
    patterns = [
        rf"\bAB\s*=\s*({NUM})\s*({UNIT_LEN})\b",
        rf"\bA\s+and\s+B\b\D{{0,60}}?(?:separated by|are|is|distance(?: of)?|apart)\D{{0,20}}?({NUM})\s*({UNIT_LEN})\b",
        rf"\bseparated by\s+(?:a\s+distance\s+of\s+)?({NUM})\s*({UNIT_LEN})\b",
        rf"\b({NUM})\s*({UNIT_LEN})\s+apart\b",
        rf"ends\s+of\s+a\s+({NUM})\s*({UNIT_LEN})\s+long\s+line\s+segment",
        rf"\bdistance\s+AB\s*=\s*({NUM})\s*({UNIT_LEN})\b",
        rf"\bdistance\s+a\s*=\s*({NUM})\s*({UNIT_LEN})\b",
        rf"\bside\s+(?:length\s+)?(?:a\s*=\s*)?({NUM})\s*({UNIT_LEN})\b",
    ]
    for pat in patterns:
        m = _find(pat, text)
        if m:
            return _parse_number(m.group(1)), m.group(2)
    return None


def _find_equilateral_side(text: str) -> tuple[float, str] | None:
    patterns = [
        rf"\bside\s+(?:length\s+)?(?:of\s+)?(?:a\s*=\s*)?({NUM})\s*({UNIT_LEN})\b",
        rf"\bwith\s+side\s+(?:a\s*=\s*)?({NUM})\s*({UNIT_LEN})\b",
    ]
    for pat in patterns:
        m = _find(pat, text)
        if m:
            return _parse_number(m.group(1)), m.group(2)
    return _find_ab_distance(text)


def _find_perp_height(text: str) -> tuple[float, str] | None:
    patterns = [
        rf"(?:distance\s*)?(?:ℓ|l)\s*=\s*({NUM})\s*({UNIT_LEN})\b",
        rf"({NUM})\s*({UNIT_LEN})\s+(?:away\s+from|from)\s+(?:AB|the\s+line\s+segment\s+AB|the\s+midpoint\s+of\s+AB)\b",
        rf"(?:distance\s+)?(?:from\s+AB|from\s+the\s+line\s+segment\s+AB)\s*(?:is|=)?\s*({NUM})\s*({UNIT_LEN})\b",
    ]
    for pat in patterns:
        m = _find(pat, text)
        if m:
            return _parse_number(m.group(1)), m.group(2)
    return None


def _extract_charges(text: str) -> list[dict[str, Any]]:
    raw: list[tuple[str, float, str]] = []
    for m in re.finditer(rf"\b(q\d+|q0|q′|q'|qo|q)\s*=\s*({NUM})\s*({UNIT_CHARGE})\b", text, flags=re.I):
        cid = m.group(1).replace("'", "′")
        raw.append((cid, _parse_number(m.group(2)), _unit(m.group(3))))

    # Handles q2 = q3 = -8e-9 C even when q1/q0 were parsed separately.
    for m in re.finditer(rf"\b(q\d+)\s*=\s*(q\d+)\s*=\s*({NUM})\s*({UNIT_CHARGE})\b", text, flags=re.I):
        value, unit = _parse_number(m.group(3)), _unit(m.group(4))
        raw.append((m.group(1), value, unit))
        raw.append((m.group(2), value, unit))

    if not raw:
        # Handles q1 = q2 = q3 = 1 μC or q1 = q2 = 5e-16 C.
        m = _find(rf"\bq1\s*=\s*q2(?:\s*=\s*q3)?\s*=\s*({NUM})\s*({UNIT_CHARGE})\b", text)
        if m:
            value, unit = _parse_number(m.group(1)), _unit(m.group(2))
            raw = [("q1", value, unit), ("q2", value, unit)]
            if "q3" in m.group(0):
                raw.append(("q3", value, unit))

    seen: set[str] = set()
    charges: list[dict[str, Any]] = []
    next_generic = 1
    for cid, value, unit in raw:
        norm_id = cid.lower()
        if norm_id in {"q′", "q'"}:
            norm_id = "q3"
        elif norm_id == "qo":
            norm_id = "q0"
        elif norm_id == "q" and any(item[0].lower() in {"q1", "q2"} for item in raw):
            norm_id = "q"
        elif norm_id == "q":
            norm_id = f"q{next_generic}"
            next_generic += 1
        if norm_id in seen:
            continue
        seen.add(norm_id)
        charges.append({"id": norm_id, "charge": _quantity(value, unit), "at": ""})
    return charges


def _assign_source_target_charges(charges: list[dict[str, Any]], target_point: str) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    source_points = ["A", "B", "C", "D"]
    for ch in charges:
        cid = ch["id"]
        ch = dict(ch)
        if cid in {"q", "q0"} or (cid == "q3" and len(charges) >= 3):
            ch["at"] = target_point
        elif cid.startswith("q") and cid[1:].isdigit():
            idx = int(cid[1:]) - 1
            ch["at"] = source_points[idx] if idx < len(source_points) else target_point
        else:
            ch["at"] = target_point
        assigned.append(ch)
    return assigned


def _query_type(low: str, has_target_charge: bool) -> str:
    if "electric field" in low or "field strength" in low or "v/m" in low or "n/c" in low:
        return "electric_field"
    return "net_force" if has_target_charge else "electric_field"


def _target_charge_id(charges: list[dict[str, Any]]) -> str | None:
    for preferred in ("q", "q0", "q3"):
        if any(ch["id"] == preferred for ch in charges):
            return preferred
    if len(charges) >= 3:
        return charges[-1]["id"]
    return None


def _schema_perpendicular_bisector(text: str, low: str) -> dict[str, Any] | None:
    if "perpendicular bisector" not in low:
        return None
    ab = _find_ab_distance(text)
    h = _find_perp_height(text)
    charges = _extract_charges(text)
    if ab is None or h is None or len(charges) < 2:
        return None

    target_point = "M" if re.search(r"\bpoint\s+M\b|\bM\b", text) else "C"
    qtype = _query_type(low, has_target_charge=len(charges) >= 3)
    assigned = _assign_source_target_charges(charges, target_point)
    query_target = target_point
    if qtype == "net_force":
        target_id = _target_charge_id(assigned)
        if target_id is None:
            return None
        query_target = target_id

    return {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": "B"}, {"id": target_point}],
        "geometry": [
            {"type": "PairwiseDistances", "points": ["A", "B"], "distances": [{"between": ["A", "B"], "value": ab[0], "unit": ab[1]}]},
            {"type": "PerpendicularBisectorPoint", "point": target_point, "between": ["A", "B"], "distance_from_segment": _length_quantity(h[0], h[1]), "orientation": "above"},
        ],
        "charges": assigned,
        "queries": [{"type": qtype, "target": query_target, "output": "magnitude", "unit": "N" if qtype == "net_force" else "V/m"}],
    }


def _schema_midpoint(text: str, low: str) -> dict[str, Any] | None:
    if "midpoint" not in low and "middle point" not in low:
        return None
    ab = _find_ab_distance(text)
    charges = _extract_charges(text)
    if ab is None or len(charges) < 2:
        return None
    point = "H" if re.search(r"\bpoint\s+H\b|\bH\b", text) else "M"
    qtype = _query_type(low, has_target_charge=len(charges) >= 3)
    assigned = _assign_source_target_charges(charges, point)
    query_target = point
    if qtype == "net_force":
        target_id = _target_charge_id(assigned)
        if target_id is None:
            return None
        query_target = target_id
    return {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": "B"}, {"id": point}],
        "geometry": [
            {"type": "PairwiseDistances", "points": ["A", "B"], "distances": [{"between": ["A", "B"], "value": ab[0], "unit": ab[1]}]},
            {"type": "Midpoint", "point": point, "between": ["A", "B"]},
        ],
        "charges": assigned,
        "queries": [{"type": qtype, "target": query_target, "output": "magnitude", "unit": "N" if qtype == "net_force" else "V/m"}],
    }


def _schema_equilateral(text: str, low: str) -> dict[str, Any] | None:
    if "equilateral" not in low:
        return None
    side = _find_equilateral_side(text)
    charges = _extract_charges(text)
    if side is None or len(charges) < 2:
        return None
    points = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
    geom = [{"type": "EquilateralTriangle", "points": ["A", "B", "C"], "side": _length_quantity(side[0], side[1]), "orientation": "above"}]
    target_point = "C"
    if "center" in low or "centroid" in low:
        points.append({"id": "O"})
        geom.append({"type": "Centroid", "point": "O", "of": ["A", "B", "C"]})
        target_point = "O"
    if target_point == "O":
        assigned = []
        for ch in charges:
            ch = dict(ch)
            if ch["id"] in {"q", "q0"}:
                ch["at"] = "O"
            elif ch["id"] == "q1":
                ch["at"] = "A"
            elif ch["id"] == "q2":
                ch["at"] = "B"
            elif ch["id"] == "q3":
                ch["at"] = "C"
            assigned.append(ch)
    else:
        assigned = _assign_source_target_charges(charges, "C")
    qtype = _query_type(low, has_target_charge=("net force" in low or "force" in low) and len(charges) >= 3)
    query_target = target_point if qtype == "electric_field" else (_target_charge_id(assigned) or assigned[-1]["id"])
    return {
        "domain": "electrostatics",
        "points": points,
        "geometry": geom,
        "charges": assigned,
        "queries": [{"type": qtype, "target": query_target, "output": "magnitude", "unit": "N" if qtype == "net_force" else "V/m"}],
    }


def _schema_straight_line_equal_spacing(text: str, low: str) -> dict[str, Any] | None:
    if not ("straight line" in low or "placed" in low and "apart on" in low and "line" in low):
        return None
    charges = _extract_charges(text)
    if len(charges) < 3:
        return None
    m = _find(rf"({NUM})\s*({UNIT_LEN})\s+apart\s+on\s+(?:a\s+)?straight\s+line", text)
    if not m:
        m = _find(rf"placed\s+({NUM})\s*({UNIT_LEN})\s+apart", text)
    if not m:
        return None
    d = (_parse_number(m.group(1)), m.group(2))
    assigned = []
    for idx, ch in enumerate(charges[:3]):
        ch = dict(ch)
        ch["at"] = ["A", "B", "C"][idx]
        assigned.append(ch)
    target_id = "q2" if "q2" in low else (_target_charge_id(assigned) or assigned[1]["id"])
    return {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "geometry": [{"type": "Collinear", "points": ["A", "B", "C"], "order": ["A", "B", "C"], "distances": [{"between": ["A", "B"], "value": d[0], "unit": d[1]}, {"between": ["B", "C"], "value": d[0], "unit": d[1]}]}],
        "charges": assigned,
        "queries": [{"type": "net_force", "target": target_id, "output": "magnitude", "unit": "N"}],
    }



def _find_force_value(text: str) -> tuple[float, str] | None:
    patterns = [
        rf"(?:force|resultant force|exert a force)\D{{0,40}}?({NUM})\s*(N)\b",
        rf"({NUM})\s*(N)\b\D{{0,40}}?(?:force|resultant)",
    ]
    for pat in patterns:
        m = _find(pat, text)
        if m:
            return _parse_number(m.group(1)), m.group(2)
    return None


def _find_two_force_magnitudes(text: str) -> list[float]:
    vals: list[float] = []
    # Prefer explicit "each of magnitude X N" as two identical vectors.
    m = _find(rf"(?:two\s+forces|two\s+electric\s+forces).*?each\s+(?:of\s+)?(?:magnitude\s+)?({NUM})\s*N", text)
    if m:
        v = _parse_number(m.group(1))
        return [v, v]
    for m in re.finditer(rf"({NUM})\s*N", text, flags=re.I):
        vals.append(_parse_number(m.group(1)))
    return vals


def _schema_inverse_coulomb_equal_charges(text: str, low: str) -> dict[str, Any] | None:
    if not ("find q" in low or "find the charge" in low or "q1 = q2 = q" in low):
        return None
    if not ("force" in low and ("separated" in low or "apart" in low or "distance" in low)):
        return None
    f = _find_force_value(text)
    r = _find_ab_distance(text)
    if f is None or r is None:
        return None
    return {
        "domain": "electrostatics",
        "queries": [{
            "type": "coulomb_equal_charge",
            "force": {"value": f[0], "unit": f[1]},
            "distance": {"value": r[0], "unit": r[1]},
            "output": "magnitude",
            "unit": "uC",
        }],
    }


def _schema_inverse_resultant_angle(text: str, low: str) -> dict[str, Any] | None:
    if not ("find the angle" in low or "angle between" in low):
        return None
    if "resultant" not in low:
        return None
    vals = _find_two_force_magnitudes(text)
    if len(vals) < 2:
        return None
    res_match = _find(rf"resultant\s+force\s+(?:is\s+)?(?:also\s+)?({NUM})\s*N", text)
    if res_match is None:
        # Last force-like number is usually the resultant in inverse-angle wording.
        all_n = [(_parse_number(m.group(1))) for m in re.finditer(rf"({NUM})\s*N", text, flags=re.I)]
        if len(all_n) < 3:
            return None
        resultant = all_n[-1]
    else:
        resultant = _parse_number(res_match.group(1))
    f1, f2 = vals[0], vals[1]
    return {
        "domain": "electrostatics",
        "vectors": [
            {"id": "F1", "magnitude": {"value": f1, "unit": "N"}},
            {"id": "F2", "magnitude": {"value": f2, "unit": "N"}},
        ],
        "queries": [{
            "type": "resultant_angle",
            "target": "vectors",
            "resultant": {"value": resultant, "unit": "N"},
            "output": "magnitude",
            "unit": "degree",
        }],
    }

def _find_distance_from_named_charge(text: str, charge_id: str) -> tuple[float, str] | None:
    patterns = [
        rf"({NUM})\s*({UNIT_LEN})\s+away\s+from\s+{re.escape(charge_id)}\b",
        rf"distance\s+from\s+{re.escape(charge_id)}\s*(?:is|=)?\s*({NUM})\s*({UNIT_LEN})\b",
    ]
    for pat in patterns:
        m = _find(pat, text)
        if m:
            return _parse_number(m.group(1)), m.group(2)
    return None


def _find_named_point_distance(text: str, a: str, b: str) -> tuple[float, str] | None:
    patterns = [
        rf"\b{re.escape(a)}{re.escape(b)}\s*=\s*({NUM})\s*({UNIT_LEN})\b",
        rf"\b{re.escape(b)}{re.escape(a)}\s*=\s*({NUM})\s*({UNIT_LEN})\b",
        rf"\b{re.escape(a)}\s*(?:is\s*)?({NUM})\s*({UNIT_LEN})\s+from\s+{re.escape(b)}\b",
        rf"\b{re.escape(b)}\s*(?:is\s*)?({NUM})\s*({UNIT_LEN})\s+from\s+{re.escape(a)}\b",
        rf"point\s+{re.escape(a)}\D{{0,20}}?({NUM})\s*({UNIT_LEN})\s+from\s+{re.escape(b)}\b",
        rf"placed\s+at\s+point\s+{re.escape(a)}\D{{0,20}}?({NUM})\s*({UNIT_LEN})\s+from\s+{re.escape(b)}\b",
        rf"({NUM})\s*({UNIT_LEN})\s+from\s+{re.escape(b)}\b",
    ]
    for pat in patterns:
        m = _find(pat, text)
        if m:
            return _parse_number(m.group(1)), m.group(2)
    return None


def _schema_point_on_segment_one_distance(text: str, low: str) -> dict[str, Any] | None:
    if not ("along the line" in low or "line connecting" in low or "positioned along" in low):
        return None
    ab = _find_ab_distance(text)
    ac = _find_distance_from_named_charge(text, "q1") or _find_named_point_distance(text, "A", "C")
    charges = _extract_charges(text)
    if ab is None or ac is None or len(charges) < 3:
        return None
    assigned: list[dict[str, Any]] = []
    for ch in charges[:3]:
        ch = dict(ch)
        ch["at"] = {"q1": "A", "q2": "B"}.get(ch["id"], "C")
        assigned.append(ch)
    return {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "geometry": [
            {"type": "PairwiseDistances", "points": ["A", "B"], "distances": [{"between": ["A", "B"], "value": ab[0], "unit": ab[1]}]},
            {"type": "PointOnLine", "point": "C", "start": "A", "end": "B", "distance_from_start": {"value": ac[0], "unit": ac[1]}, "direction": "toward_end"},
        ],
        "charges": assigned,
        "queries": [{"type": "net_force", "target": _target_charge_id(assigned) or assigned[-1]["id"], "output": "magnitude", "unit": "N"}],
    }


def _schema_opposite_sides_target_centered(text: str, low: str) -> dict[str, Any] | None:
    if "opposite sides" not in low:
        return None
    charges = _extract_charges(text)
    if len(charges) < 3:
        two_sources = _find(rf"two\s+([+-]?{NUM})\s*({UNIT_CHARGE})\s+charges", text)
        target = next((ch for ch in charges if ch["id"] in {"q", "q0"}), None)
        if target is None and len(charges) == 1:
            target = dict(charges[0])
            target["id"] = "q"
        if two_sources is not None and target is not None:
            v = _parse_number(two_sources.group(1))
            u = _unit(two_sources.group(2))
            charges = [target, {"id": "q1", "charge": _quantity(v, u), "at": ""}, {"id": "q2", "charge": _quantity(v, u), "at": ""}]
    if len(charges) < 3:
        return None
    nums = list(re.finditer(rf"({NUM})\s*({UNIT_LEN})", text, flags=re.I))
    if len(nums) < 2:
        return None
    d1 = (_parse_number(nums[-2].group(1)), nums[-2].group(2))
    d2 = (_parse_number(nums[-1].group(1)), nums[-1].group(2))
    target = "Q"
    assigned: list[dict[str, Any]] = []
    # The target charge is the one named q/q0; sources go to A/B.
    src_points = iter(["A", "B"])
    target_id = None
    for ch in charges:
        ch = dict(ch)
        if target_id is None and (ch["id"] in {"q", "q0"} or len(assigned) == 0):
            ch["at"] = target
            target_id = ch["id"]
        else:
            ch["at"] = next(src_points, "B")
        assigned.append(ch)
    if target_id is None:
        target_id = assigned[0]["id"]
        assigned[0]["at"] = target
    return {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": target}, {"id": "B"}],
        "geometry": [{"type": "Collinear", "points": ["A", target, "B"], "order": ["A", target, "B"], "distances": [{"between": ["A", target], "value": d1[0], "unit": d1[1]}, {"between": [target, "B"], "value": d2[0], "unit": d2[1]}]}],
        "charges": assigned,
        "queries": [{"type": "net_force", "target": target_id, "output": "magnitude", "unit": "N"}],
    }


def _schema_three_point_pairwise_from_text(text: str, low: str) -> dict[str, Any] | None:
    # Handles "q0 at M, 4 cm from A and 12 cm from B" when the LLM drops AM/BM.
    if not ("from a" in low and "from b" in low and ("point m" in low or "placed at m" in low or "placed at point m" in low)):
        return None
    ab = _find_ab_distance(text)
    am = _find_named_point_distance(text, "M", "A")
    bm = _find_named_point_distance(text, "M", "B")
    charges = _extract_charges(text)
    if ab is None or am is None or bm is None or len(charges) < 2:
        return None
    assigned = _assign_source_target_charges(charges, "M")
    target_id = _target_charge_id(assigned) or assigned[-1]["id"]
    return {
        "domain": "electrostatics",
        "points": [{"id": "A"}, {"id": "B"}, {"id": "M"}],
        "geometry": [{"type": "PairwiseDistances", "points": ["A", "B", "M"], "distances": [
            {"between": ["A", "B"], "value": ab[0], "unit": ab[1]},
            {"between": ["A", "M"], "value": am[0], "unit": am[1]},
            {"between": ["B", "M"], "value": bm[0], "unit": bm[1]},
        ], "orientation": "above"}],
        "charges": assigned,
        "queries": [{"type": "net_force", "target": target_id, "output": "magnitude", "unit": "N"}],
    }

def extract_electrostatics_schema_from_text(problem: str) -> dict[str, Any] | None:
    text = _norm(problem)
    low = text.lower()
    for builder in (
        _schema_inverse_coulomb_equal_charges,
        _schema_inverse_resultant_angle,
        _schema_opposite_sides_target_centered,
        _schema_point_on_segment_one_distance,
        _schema_three_point_pairwise_from_text,
        _schema_perpendicular_bisector,
        _schema_midpoint,
        _schema_equilateral,
        _schema_straight_line_equal_spacing,
    ):
        schema = builder(text, low)
        if schema is not None:
            return schema
    return None
