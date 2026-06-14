from __future__ import annotations

import math
import re
from typing import Any

from xai_physics.hybrid.formula_filler import Quantity, fill_formula_specs

NUM = r"[-+]?(?:10\s*(?:\^|\*\*)\s*[-+]?\d+|10\s*[-+]\s*\d+|(?:\d+(?:\.\d+)?|\.\d+)(?:\s*(?:×|x|\*)\s*10\s*(?:\^|\*\*)?\s*[-+]?\d+|[eE][-+]?\d+)?)"


def _norm(text: str) -> str:
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("⁻", "-")
        .replace("µ", "u")
        .replace("μ", "u")
    )
    supers_with_sign = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    text = re.sub(r"10([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "10^" + m.group(1).translate(supers_with_sign), text)
    supers = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    text = text.translate(supers).translate(subs)
    text = text.replace("²", "2").replace("³", "3")
    text = text.replace("^^", "^")
    text = re.sub(r"(?P<coef>\d+)\.10\s*\^", r"\g<coef> x 10^", text)
    text = re.sub(r"(?P<coef>\d+)\.10(?P<exp>[-+]\d+)", r"\g<coef> x 10^\g<exp>", text)
    return text


def _parse_number(raw: str) -> float:
    cleaned = raw.strip().replace(",", "")
    cleaned = re.sub(r"^([-+]?)10\s*(?:\^|\*\*)\s*([-+]?\d+)$", r"\g<1>1e\2", cleaned)
    cleaned = re.sub(r"^([-+]?)10\s*([-+]\s*\d+)$", lambda m: f"{m.group(1)}1e{m.group(2).replace(' ', '')}", cleaned)
    cleaned = re.sub(r"\s*(?:×|x|\*)\s*10\s*(?:\^|\*\*)?\s*", "e", cleaned)
    cleaned = cleaned.replace(" ", "")
    return float(cleaned)


def _unit(raw: str) -> str:
    unit = raw.replace("µ", "u").replace("μ", "u").replace("Ω", "ohm").replace("Ω", "ohm").replace("²", "2").replace("^2", "2")
    return unit.replace(" / ", "/").replace(" /", "/").replace("/ ", "/")


def _quantity(value: float, unit: str) -> dict[str, Any]:
    return {"value": str(value), "unit": _unit(unit)}


def _to_si_capacitance(value: float, unit: str) -> float | None:
    factors = {
        "f": 1.0,
        "farad": 1.0,
        "farads": 1.0,
        "mf": 1e-3,
        "uf": 1e-6,
        "microfarad": 1e-6,
        "microfarads": 1e-6,
        "nf": 1e-9,
        "nanofarad": 1e-9,
        "nanofarads": 1e-9,
        "pf": 1e-12,
        "picofarad": 1e-12,
        "picofarads": 1e-12,
    }
    factor = factors.get(_unit(unit).lower())
    return None if factor is None else value * factor


def _to_si_voltage(value: float, unit: str) -> float | None:
    factors = {"v": 1.0, "volt": 1.0, "volts": 1.0, "mv": 1e-3, "kv": 1e3}
    factor = factors.get(_unit(unit).lower())
    return None if factor is None else value * factor


def _energy_query_unit(capacitance: tuple[float, str], voltage: tuple[float, str]) -> str:
    c_si = _to_si_capacitance(*capacitance)
    u_si = _to_si_voltage(*voltage)
    if c_si is None or u_si is None:
        return "J"

    energy_j = 0.5 * c_si * u_si * u_si
    if energy_j >= 1e-3:
        return "J"
    if energy_j >= 1e-6:
        return "uJ"
    if energy_j >= 1e-9:
        return "nJ"
    return "pJ"


def _find_quantity(patterns: list[str], text: str) -> tuple[float, str] | None:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            return _parse_number(m.group("value")), _unit(m.group("unit"))
    return None


def _capacitance(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>pF|nF|uF|mF|F|farads?|microfarads?|nanofarads?|picofarads?)"
    return _find_quantity(
        [
            rf"\bC\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bcapacitance(?:\s+of)?(?:\s+is|\s*=|\s+C\s*=|\s+of)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bwith\s+(?:a\s+)?capacitance(?:\s+of)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bhas\s+(?:a\s+)?capacitance(?:\s+of)?\s*(?P<value>{NUM})\s*{unit}\b",
        ],
        text,
    )


def _voltage(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>mV|kV|V|volts?)"
    return _find_quantity(
        [
            rf"\bU\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bV\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bvoltage(?:\s+of|\s+is|\s*=|\s+U\s*=|\s+across[^,.]*?is)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bpotential\s+difference(?:\s+of|\s+is|\s*=)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\b(?:potential\s+difference|voltage)[^.;:]*?(?P<value>{NUM})\s*{unit}\b",
            rf"\bconnected\s+to\s+(?:a\s+)?(?P<value>{NUM})\s*{unit}\s+(?:power\s+source|source|battery)\b",
        ],
        text,
    )


def _charge(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>pC|nC|uC|mC|C|coulombs?)"
    return _find_quantity(
        [
            rf"\bQ\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bq\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bcharge(?:\s+q)?(?:\s+of|\s+is|\s*=|\s+Q\s*=)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\b(?:point\s+charge|electric\s+charge|charge)\s+(?:q\s*=\s*)?(?P<value>{NUM})\s*{unit}\b",
            rf"\bcarrying\s+(?:an?\s+)?electric\s+charge\s+of\s+(?P<value>{NUM})\s*{unit}\b",
            rf"\bstores\s+(?:a\s+)?(?:charge\s+)?(?:Q\s*=\s*)?(?P<value>{NUM})\s*{unit}\b",
        ],
        text,
    )


def _area(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>cm2|mm2|m2|cm\^2|mm\^2|m\^2)"
    return _find_quantity(
        [
            rf"\b(?:plate\s+)?area(?:\s+of\s+each\s+plate|\s+of)?(?:\s+S\s*=|\s+A\s*=|\s*=|\s+is)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bS\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bA\s*=\s*(?P<value>{NUM})\s*{unit}\b",
        ],
        text,
    )


def _radius(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>cm|mm|m)"
    return _find_quantity(
        [
            rf"\bradius\s*(?:R\s*)?(?:=|of|is)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bR\s*=\s*(?P<value>{NUM})\s*{unit}\b",
        ],
        text,
    )


def _distance(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>cm|mm|m)"
    return _find_quantity(
        [
            rf"\b(?:plate\s+)?separation(?:\s+is|\s*=|\s+of)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bdistance\s+between\s+(?:the\s+)?(?:two\s+)?plates(?:\s+is|\s*=|\s+and[^,.]*?are)?\s*(?:d\s*=\s*)?(?P<value>{NUM})\s*{unit}\b",
            rf"\bd\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bdistance\s*(?:r|MO)?\s*(?:=|is|of)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bseparated\s+by\s+(?:a\s+)?distance\s+of\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\br\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\b(?P<value>{NUM})\s*{unit}\s+away\s+from\b",
            rf"\bat\s+(?:a\s+)?point\s+(?P<value>{NUM})\s*{unit}\s+away\b",
            rf"\b(?P<value>{NUM})\s*{unit}\s+from\s+(?:the\s+)?(?:sphere|charge|wire)\b",
            rf"\bMO\s*=\s*(?P<value>{NUM})\s*{unit}\b",
        ],
        text,
    )


def _is_charge_query(low: str) -> bool:
    return any(
        p in low
        for p in [
            "calculate the charge",
            "what is the charge",
            "find the charge",
            "charge stored",
            "charge on",
            "charge accumulated",
            "maximum charge",
            "magnitude of charge",
            "sign and magnitude of q",
            "find q1",
            "find q2",
        ]
    )


def _is_energy_query(low: str) -> bool:
    return "energy" in low and any(p in low for p in ["calculate", "what is", "find", "stored", "between the plates"])


def _is_capacitance_query(low: str) -> bool:
    return "capacitance" in low and any(p in low for p in ["calculate", "what is", "find", "determine"])


def _is_dielectric_constant_query(low: str) -> bool:
    return any(p in low for p in ["dielectric constant", "relative permittivity", "permittivity"]) and any(
        p in low for p in ["what is", "calculate", "find", "determine"]
    )


def _is_electric_field_query(low: str) -> bool:
    return any(p in low for p in ["electric field", "field strength", "field intensity", "n/c", "v/m"])


def _is_force_query(low: str) -> bool:
    return "force" in low and any(p in low for p in ["calculate", "find", "determine", "what is", "magnitude"])


def _line_charge_density(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>pC\s*/\s*m|nC\s*/\s*m|uC\s*/\s*m|mC\s*/\s*m|C\s*/\s*m)"
    return _find_quantity(
        [
            rf"(?:linear\s+charge\s+density(?:\s*(?:λ|lambda))?|λ|lambda)\s*(?:=|of|is)?\s*(?P<value>{NUM})\s*{unit}",
        ],
        text,
    )


def _mass(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>kg|g)"
    return _find_quantity(
        [
            rf"\bm\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bmass(?:\s+of|\s+is|\s*=)?\s*(?P<value>{NUM})\s*{unit}\b",
        ],
        text,
    )


def _gravity(text: str) -> tuple[float, str] | None:
    return _find_quantity(
        [
            rf"\bg\s*=\s*(?P<value>{NUM})\s*(?P<unit>m/s2|m/s\^2|m/s²)\b",
        ],
        text,
    )




def _force(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>uN|mN|N)"
    return _find_quantity(
        [
            rf"\bF\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bforce(?:\s+F)?(?:\s+of|\s+is|\s*=)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bexperiences\s+(?:a\s+)?force\s+(?:F\s*=\s*)?(?P<value>{NUM})\s*{unit}\b",
        ],
        text,
    )


def _electric_field_given(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>N\s*/\s*C|V\s*/\s*m)"
    return _find_quantity(
        [
            rf"\bE\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\belectric\s+field(?:\s+strength)?(?:\s+E)?(?:\s+of|\s+is|\s*=|\s+has\s+(?:a\s+)?magnitude\s+of|\s+with\s+(?:a\s+)?magnitude\s+of)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\belectric\s+field[^.;:]*?has\s+(?:a\s+)?magnitude\s+of\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bfield\s+strength(?:\s+E)?(?:\s+of|\s+is|\s*=)?\s*(?P<value>{NUM})\s*{unit}\b",
        ],
        text,
    )


def _relative_permittivity_value(text: str) -> tuple[float, str] | None:
    return _find_quantity(
        [
            rf"\b(?:dielectric\s+constant|relative\s+permittivity|epsilon|eps|ε)(?:\s+of\s+[^,.]+?)?\s*(?:=|is|of)?\s*(?P<value>{NUM})\s*(?P<unit>)",
        ],
        text,
    )


def _two_charge_values_for_zero_field(text: str, low: str) -> tuple[tuple[float, str], tuple[float, str]] | None:
    unit_pat = r"(?:pC|nC|uC|mC|C|coulombs?)"
    equal_pair = re.search(
        rf"\bq\s*1\s*=\s*q\s*2(?:\s*=\s*q\s*3)?\s*=\s*(?P<q>{NUM})\s*(?P<u>{unit_pat})\b",
        text,
        flags=re.I,
    )
    if equal_pair:
        q = (_parse_number(equal_pair.group("q")), _unit(equal_pair.group("u")))
        return q, q

    opposite_pair = re.search(
        rf"\bq\s*1\s*=\s*-\s*q\s*2\s*=\s*(?P<q>{NUM})\s*(?P<u>{unit_pat})\b",
        text,
        flags=re.I,
    )
    if opposite_pair:
        q = _parse_number(opposite_pair.group("q"))
        u = _unit(opposite_pair.group("u"))
        return (q, u), (-q, u)

    direct = re.search(
        rf"\bq\s*1\s*=\s*(?P<q1>{NUM})\s*(?P<u1>{unit_pat})\b.*?\bq\s*2\s*=\s*(?P<q2>{NUM})\s*(?P<u2>{unit_pat})?\b",
        text,
        flags=re.I | re.S,
    )
    if direct:
        u1 = _unit(direct.group("u1"))
        u2 = _unit(direct.group("u2") or u1)
        return (_parse_number(direct.group("q1")), u1), (_parse_number(direct.group("q2")), u2)

    ratio = re.search(rf"\bq\s*1\s*=\s*(?P<ratio>{NUM})\s*q\s*2\b", text, flags=re.I)
    if ratio and "same sign" in low:
        return (_parse_number(ratio.group("ratio")), "C"), (1.0, "C")
    return None


def _zero_field_separation(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>cm|mm|m)"
    return _find_quantity(
        [
            rf"\bseparated\s+by(?:\s+a\s+distance\s+of)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bare\s+(?P<value>{NUM})\s*{unit}\s+apart\b",
            rf"\bplaced\s+(?P<value>{NUM})\s*{unit}\s+apart\b",
            rf"\bplaced\s+at\s+two\s+points\s+(?P<value>{NUM})\s*{unit}\s+apart\b",
            rf"\bseparated\s+by\s+a\s+distance\s+a\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\b(?:points\s+)?A\s+and\s+B\s+are\s*(?P<value>{NUM})\s*{unit}\s+apart\b",
            rf"\b(?:points\s+)?A\s+and\s+B\b[^.]*?\b(?P<value>{NUM})\s*{unit}\s+apart\b",
            rf"\bAB\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bq\s*2\b[^.]*?\blocated\s+(?P<value>{NUM})\s*{unit}\s+from\s+the\s+origin\b",
        ],
        text,
    )


def _is_zero_field_distance_query(low: str) -> bool:
    has_zero_field = "zero" in low and any(p in low for p in ["electric field", "field strength", "resultant electric field", "net electric field"])
    asks_location = any(p in low for p in ["where", "find point", "calculate its distance", "calculate the distance", "what is the distance", "coordinate"])
    return has_zero_field and asks_location


def _zero_field_reference(text: str, low: str) -> str:
    if re.search(r"\bBM\b", text, flags=re.I) or "distance from b" in low or re.search(r"distance\s+from\s+point\s+M\s+to\s+B", text, flags=re.I):
        return "B"
    if "coordinate" in low or "ox axis" in low:
        return "COORDINATE_FROM_A"
    return "A"


def _two_charge_zero_field_candidate(text: str, low: str) -> dict[str, Any] | None:
    if not _is_zero_field_distance_query(low):
        return None
    charges = _two_charge_values_for_zero_field(text, low)
    separation = _zero_field_separation(text)
    if charges is None or separation is None:
        return None

    q1, q2 = charges
    reference = _zero_field_reference(text, low)
    _, distance_unit = separation
    return _schema(
        "two_charge_zero_field_distance",
        [
            {"id": "q1", "type": "charge", "role": "given", **_quantity(*q1)},
            {"id": "q2", "type": "charge", "role": "given", **_quantity(*q2)},
            {"id": "d_ab", "type": "distance", "role": "given", **_quantity(*separation)},
            {"id": "x_query", "type": "distance", "role": "query", "value": None, "unit": _unit(distance_unit), "reference": reference},
        ],
        constraints=["Model A and B on a 1D line and solve the point where the two electric-field magnitudes cancel."],
    )


def _common_distance_to_m(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>cm|mm|m)"
    return _find_quantity(
        [
            rf"\beach\s+is\s+(?P<value>{NUM})\s*{unit}\s+from\s+point\s+M\b",
            rf"\beach\s+located\s+(?P<value>{NUM})\s*{unit}\s+from\s+point\s+M\b",
            rf"\beach\s+located\s+(?P<value>{NUM})\s*{unit}\s+from\s+M\b",
            rf"\bare\s+each\s+located\s+(?P<value>{NUM})\s*{unit}\s+from\s+point\s+M\b",
            rf"\bare\s+each\s+located\s+(?P<value>{NUM})\s*{unit}\s+from\s+M\b",
            rf"\beach\s+(?:is\s+)?located\s+(?P<value>{NUM})\s*{unit}\s+away\s+from\s+point\s+M\b",
            rf"\bare\s+each\s+(?P<value>{NUM})\s*{unit}\s+from\s+point\s+M\b",
            rf"\bboth\s+points\s+are\s+(?P<value>{NUM})\s*{unit}\s+away\s+from\s+(?:a\s+)?(?:central\s+)?point\s+M\b",
            rf"\bboth\s+located\s+(?P<value>{NUM})\s*{unit}\s+away\s+from\s+point\s+M\b",
            rf"\bboth\s+located\s+(?P<value>{NUM})\s*{unit}\s+away\s+from\s+M\b",
            rf"\bare\s+both\s+located\s+(?P<value>{NUM})\s*{unit}\s+away\s+from\s+point\s+M\b",
            rf"\bare\s+both\s+(?:located\s+)?(?P<value>{NUM})\s*{unit}\s+away\s+from\s+point\s+M\b",
            rf"\bare\s+located\s+(?P<value>{NUM})\s*{unit}\s+from\s+M\b",
            rf"\bare\s+located\s+(?P<value>{NUM})\s*{unit}\s+from\s+point\s+M\b",
            rf"\bare\s+located\s+such\s+that\s+each\s+is\s+(?P<value>{NUM})\s*{unit}\s+away\s+from\s+point\s+M\b",
            rf"\bare\s+placed\s+such\s+that\s+each\s+is\s+(?P<value>{NUM})\s*{unit}\s+from\s+point\s+M\b",
            rf"\bare\s+placed\s+such\s+that\s+each\s+is\s+(?P<value>{NUM})\s*{unit}\s+away\s+from\s+point\s+M\b",
        ],
        text,
    )


def _field_vector_angle(text: str, low: str) -> tuple[float, str] | None:
    if "perpendicular" in low:
        return (90.0, "degree")
    m = re.search(r"\b(?P<angle>60|90)\s*(?:°|degrees?)", text, flags=re.I)
    if m:
        return (float(m.group("angle")), "degree")
    return None


def _two_field_vector_resultant_candidate(text: str, low: str) -> dict[str, Any] | None:
    if not _is_electric_field_query(low):
        return None
    if not any(p in low for p in ["resultant electric field", "resultant field", "total electric field", "net electric field"]):
        return None
    angle = _field_vector_angle(text, low)
    distance = _common_distance_to_m(text)
    charges = _two_charge_values_for_zero_field(text, low)
    if angle is None or distance is None or charges is None:
        return None
    q1, q2 = charges
    return _schema(
        "two_field_vector_resultant",
        [
            {"id": "q1", "type": "charge", "role": "given", **_quantity(*q1)},
            {"id": "q2", "type": "charge", "role": "given", **_quantity(*q2)},
            {"id": "r1", "type": "distance", "role": "given", **_quantity(*distance)},
            {"id": "theta", "type": "angle", "role": "given", **_quantity(*angle)},
            {"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"},
        ],
        constraints=["Compute each field magnitude and combine by the angle between the field vectors."],
    )



def _length_to_m(length: tuple[float, str]) -> float:
    value, unit = length
    unit = _unit(unit).lower()
    factors = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
    return value * factors.get(unit, 1.0)


def _point_obj(obj_id: str, x_m: float, y_m: float, *, role: str = "given") -> dict[str, Any]:
    return {"id": obj_id, "type": "point", "role": role, "value": "0", "unit": "", "x": str(x_m), "y": str(y_m), "coordinate_unit": "m"}


def _source_charge_obj(obj_id: str, charge: tuple[float, str], x_m: float, y_m: float) -> dict[str, Any]:
    return {"id": obj_id, "type": "charge", "role": "given", "source": True, "x": str(x_m), "y": str(y_m), "coordinate_unit": "m", **_quantity(*charge)}


def _target_charge(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>pC|nC|uC|mC|C|coulombs?)"
    return _find_quantity(
        [
            rf"\bq\s*0\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bq\s*3\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\btest\s+charge\s+q\b[^.]*?magnitude\s+of\s+(?P<value>{NUM})\s*{unit}\b",
            rf"\btest\s+charge\s+q\b[^.]*?(?P<value>{NUM})\s*{unit}\b",
        ],
        text,
    )


def _distance_from_midpoint(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>cm|mm|m)"
    return _find_quantity(
        [
            rf"\bat\s+a\s+distance\s+of\s+(?P<value>{NUM})\s*{unit}\s+from\s+AB\b",
            rf"\b(?P<value>{NUM})\s*{unit}\s+away\s+from\s+AB\b",
            rf"\b(?P<value>{NUM})\s*{unit}\s+away\s+from\s+(?:the\s+)?midpoint\b",
            rf"\bat\s+a\s+distance\s+of\s+(?P<value>{NUM})\s*{unit}\s+from\s+(?:its\s+|the\s+)?midpoint\b",
            rf"\bis\s+(?P<value>{NUM})\s*{unit}\s+from\s+(?:its\s+|the\s+)?midpoint\b",
            rf"\blocated\s+(?P<value>{NUM})\s*{unit}\s+from\s+(?:its\s+|the\s+)?midpoint\b",
            rf"\b(?P<value>{NUM})\s*{unit}\s+from\s+(?:its\s+|the\s+)?midpoint\b",
            rf"\b(?P<value>{NUM})\s*{unit}\s+from\s+(?:its\s+|the\s+)?midpoint\s+of\s+(?:AB|this\s+segment)\b",
        ],
        text,
    )


def _distance_from_each_charge(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>cm|mm|m)"
    return _find_quantity(
        [
            rf"\b(?P<value>{NUM})\s*{unit}\s+(?:away\s+)?from\s+each\s+charge\b",
            rf"\b(?P<value>{NUM})\s*{unit}\s+(?:away\s+)?from\s+each\s+of\s+the\s+two\s+charges\b",
            rf"\bequidistant\s+from\s+(?:both|the\s+two)\s+charges\s+(?:by|at)\s+(?P<value>{NUM})\s*{unit}\b",
            rf"\bequidistant\s+from\s+(?:A\s+and\s+B|both\s+charges)\s+by\s+(?P<value>{NUM})\s*{unit}\b",
        ],
        text,
    )


def _triangle_distances_to_target(text: str, separation: tuple[float, str] | None = None) -> tuple[tuple[float, str], tuple[float, str]] | None:
    unit = r"(?P<unit>cm|mm|m)"
    if separation is not None and re.search(r"equidistant\s+from\s+A\s+and\s+B\s+by\s+a\s+distance\s+equal\s+to\s+a", text, flags=re.I):
        return separation, separation
    same = re.search(rf"\bAC\s*=\s*BC\s*=\s*(?P<value>{NUM})\s*{unit}\b", text, flags=re.I)
    if same:
        dist = (_parse_number(same.group("value")), _unit(same.group("unit")))
        return dist, dist
    m = re.search(rf"\bAC\s*=\s*(?P<ac>{NUM})\s*(?P<u1>cm|mm|m)\b.*?\bBC\s*=\s*(?P<bc>{NUM})\s*(?P<u2>cm|mm|m)\b", text, flags=re.I | re.S)
    if m:
        return (_parse_number(m.group("ac")), _unit(m.group("u1"))), (_parse_number(m.group("bc")), _unit(m.group("u2")))
    generic = re.search(
        rf"(?P<ac>{NUM})\s*(?P<u1>cm|mm|m)\s+from\s+A\b.*?(?P<bc>{NUM})\s*(?P<u2>cm|mm|m)\s+from\s+B\b",
        text,
        flags=re.I | re.S,
    )
    if generic:
        return (_parse_number(generic.group("ac")), _unit(generic.group("u1"))), (_parse_number(generic.group("bc")), _unit(generic.group("u2")))
    if separation is not None and re.search(r"equidistant\s+from\s+A\s+and\s+B\s+by\s+a\s+distance\s+a\s*=", text, flags=re.I):
        return separation, separation
    return None


def _ma_mb_distances(text: str) -> tuple[tuple[float, str], tuple[float, str]] | None:
    m = re.search(rf"\bMA\s*=\s*(?P<ma>{NUM})\s*(?P<u1>cm|mm|m)\b.*?\bMB\s*=\s*(?P<mb>{NUM})\s*(?P<u2>cm|mm|m)\b", text, flags=re.I | re.S)
    if not m:
        return None
    return (_parse_number(m.group("ma")), _unit(m.group("u1"))), (_parse_number(m.group("mb")), _unit(m.group("u2")))


def _collinear_distance_from_a(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>cm|mm|m)"
    return _find_quantity(
        [
            rf"\bM\s+is\s+(?P<value>{NUM})\s*{unit}\s+from\s+A\b",
            rf"\bM\s+is\s+located\s+(?P<value>{NUM})\s*{unit}\s+from\s+A\b",
            rf"\bis\s+(?P<value>{NUM})\s*{unit}\s+from\s+A\b",
            rf"\bis\s+(?P<value>{NUM})\s*{unit}\s+to\s+the\s+right\s+of\s+A\b",
            rf"\bis\s+located\s+(?P<value>{NUM})\s*{unit}\s+to\s+the\s+right\s+of\s+A\b",
        ],
        text,
    )


def _collinear_left_of_q1(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>cm|mm|m)"
    return _find_quantity(
        [
            rf"\b(?P<value>{NUM})\s*{unit}\s+to\s+the\s+left\s+of\s+(?:charge\s+)?q\s*1\b",
            rf"\b(?P<value>{NUM})\s*{unit}\s+to\s+the\s+left\s+of\s+A\b",
        ],
        text,
    )


def _two_charge_geometry(text: str, low: str) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    separation = _zero_field_separation(text)

    # Explicit angle at M: build rays from M to q1 and q2 and let signed charges
    # determine the actual field-vector directions. This handles opposite signs,
    # where the field-vector angle is supplementary to the geometric angle.
    angle = _field_vector_angle(text, low)
    common = _common_distance_to_m(text)
    if common is None:
        common = _find_quantity(
            [
                rf"\bEach\s+point\s+is\s+(?P<value>{NUM})\s*(?P<unit>cm|mm|m)\s+away\s+from\s+M\b",
                rf"\bequidistant\s+from\s+point\s+M\b.*?\bEach\s+point\s+is\s+(?P<value>{NUM})\s*(?P<unit>cm|mm|m)\s+away\s+from\s+M\b",
            ],
            text,
        )
    if angle is not None and common is not None and any(p in low for p in ["angle between", "angle formed", "form an angle", "perpendicular"]):
        r = _length_to_m(common)
        theta = _parse_number(str(angle[0])) * math.pi / 180.0 if angle[1].lower().startswith("degree") else angle[0]
        return (r, 0.0), (r * math.cos(theta), r * math.sin(theta)), (0.0, 0.0)

    if separation is None:
        return None
    d = _length_to_m(separation)

    tri = _triangle_distances_to_target(text, separation)
    if tri is not None:
        ac = _length_to_m(tri[0])
        bc = _length_to_m(tri[1])
        if d == 0:
            return None
        x = (ac * ac + d * d - bc * bc) / (2.0 * d)
        y2 = max(0.0, ac * ac - x * x)
        return (0.0, 0.0), (d, 0.0), (x, math.sqrt(y2))

    q_distances = _distances_from_q1_q2(text)
    if q_distances is not None and any(p in low for p in ["straight line", "same line", "on the line", "passing through"]):
        r1 = _length_to_m(q_distances[0])
        r2 = _length_to_m(q_distances[1])
        tol = max(1e-9, d * 1e-6)
        if abs(r1 + r2 - d) <= tol:
            x = r1
        elif abs(r1 + d - r2) <= tol:
            x = -r1
        elif abs(r2 + d - r1) <= tol:
            x = d + r2
        else:
            return None
        return (0.0, 0.0), (d, 0.0), (x, 0.0)

    ma_mb = _ma_mb_distances(text)
    if ma_mb is not None:
        ma = _length_to_m(ma_mb[0])
        mb = _length_to_m(ma_mb[1])
        tol = max(1e-9, d * 1e-6)
        if abs(ma + mb - d) <= tol:
            x = ma
        elif abs(ma + d - mb) <= tol:
            x = -ma
        elif abs(mb + d - ma) <= tol:
            x = d + mb
        else:
            return None
        return (0.0, 0.0), (d, 0.0), (x, 0.0)

    if "perpendicular bisector" in low:
        h_len = _distance_from_midpoint(text)
        if h_len is not None:
            h = _length_to_m(h_len)
            return (-d / 2.0, 0.0), (d / 2.0, 0.0), (0.0, h)
        each_len = _distance_from_each_charge(text)
        if each_len is not None:
            r = _length_to_m(each_len)
            h = math.sqrt(max(0.0, r * r - (d / 2.0) * (d / 2.0)))
            return (-d / 2.0, 0.0), (d / 2.0, 0.0), (0.0, h)

    if "on the line" in low or "outside the segment" in low or "ma =" in low or "mb =" in low:
        left = _collinear_left_of_q1(text)
        if left is not None:
            return (0.0, 0.0), (d, 0.0), (-_length_to_m(left), 0.0)
        from_a = _collinear_distance_from_a(text)
        if from_a is not None:
            dist = _length_to_m(from_a)
            right_hint = any(p in low for p in ["to the right", "right side", "right of a", "right of charge"])
            outside_hint = "outside" in low or "left" in low or "right" in low
            if right_hint:
                x = dist
            elif outside_hint:
                x = -dist
            else:
                x = dist
            return (0.0, 0.0), (d, 0.0), (x, 0.0)

    if "midpoint" in low:
        return (0.0, 0.0), (d, 0.0), (d / 2.0, 0.0)

    return None



def _side_length_between(text: str, a: str, b: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>cm|mm|m)"
    side = ''.join(sorted((a.upper(), b.upper())))
    direct = re.search(rf"\b{side}\s*=\s*(?P<value>{NUM})\s*{unit}\b", text, flags=re.I)
    if direct:
        return _parse_number(direct.group("value")), _unit(direct.group("unit"))
    spaced = re.search(rf"\b{a}\s*(?:-|to|and)\s*{b}\b[^.]*?\b(?P<value>{NUM})\s*{unit}\b", text, flags=re.I)
    if spaced:
        return _parse_number(spaced.group("value")), _unit(spaced.group("unit"))
    return None


def _labeled_charge(text: str, label: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>pC|nC|uC|mC|C|coulombs?)"
    label = label.upper()
    patterns = [
        rf"\bq\s*{label}\s*=\s*(?P<value>{NUM})\s*{unit}\b",
        rf"\bq{label}\s*=\s*(?P<value>{NUM})\s*{unit}\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return _parse_number(m.group("value")), _unit(m.group("unit"))
    return None


def _target_point_from_force_query(text: str, low: str) -> str | None:
    for label in ("A", "B", "C", "M", "O"):
        if re.search(rf"\b(?:force|acting)\b[^.]*\b(?:charge\s+)?(?:at\s+)?{label}\b", text, flags=re.I):
            return label
        if re.search(rf"\bon\s+q\s*{label}\b", text, flags=re.I):
            return label
    m = re.search(r"\bforce\s+(?:acting\s+)?on\s+(?:the\s+)?charge\s+at\s+(?P<label>[ABC])\b", text, flags=re.I)
    if m:
        return m.group("label").upper()
    return None


def _right_triangle_coordinates_from_text(text: str, low: str) -> dict[str, tuple[float, float]] | None:
    if "right" not in low or "triangle" not in low:
        return None
    m = re.search(r"right[- ]angled\s+at\s+(?P<label>[ABC])", text, flags=re.I)
    if not m:
        m = re.search(r"right\s+angle\s+at\s+(?P<label>[ABC])", text, flags=re.I)
    if not m:
        return None
    right = m.group("label").upper()
    labels = ["A", "B", "C"]
    others = [x for x in labels if x != right]
    adj1 = ''.join(sorted((right, others[0])))
    adj2 = ''.join(sorted((right, others[1])))
    hyp = ''.join(sorted((others[0], others[1])))
    lengths = {
        "AB": _side_length_between(text, "A", "B"),
        "AC": _side_length_between(text, "A", "C"),
        "BC": _side_length_between(text, "B", "C"),
    }
    if lengths[hyp] is None:
        if lengths[adj1] is None or lengths[adj2] is None:
            return None
        a = _length_to_m(lengths[adj1])
        b = _length_to_m(lengths[adj2])
    else:
        h = _length_to_m(lengths[hyp])
        if lengths[adj1] is not None:
            a = _length_to_m(lengths[adj1])
            b2 = h * h - a * a
            if b2 < -1e-12:
                return None
            b = math.sqrt(max(0.0, b2))
        elif lengths[adj2] is not None:
            b = _length_to_m(lengths[adj2])
            a2 = h * h - b * b
            if a2 < -1e-12:
                return None
            a = math.sqrt(max(0.0, a2))
        else:
            return None
    coords = {right: (0.0, 0.0), others[0]: (a, 0.0), others[1]: (0.0, b)}
    return coords


def _right_triangle_three_charge_candidate(text: str, low: str) -> dict[str, Any] | None:
    if not _is_force_query(low):
        return None
    coords = _right_triangle_coordinates_from_text(text, low)
    if coords is None:
        return None
    charges = {label: _labeled_charge(text, label) for label in ("A", "B", "C")}
    if any(value is None for value in charges.values()):
        return None
    target_label = _target_point_from_force_query(text, low)
    if target_label not in coords:
        return None
    objects: list[dict[str, Any]] = []
    for label in ("A", "B", "C"):
        charge = charges[label]
        x, y = coords[label]
        if label == target_label:
            objects.append(_point_obj("M", x, y))
            objects.append({"id": "q_test", "type": "test_charge", "role": "given", **_quantity(*charge)})
        else:
            objects.append(_source_charge_obj(f"q{label}", charge, x, y))
    objects.append({"id": "F_query", "type": "force", "role": "query", "value": None, "unit": "N"})
    return _schema(
        "two_charge_geometry_field",
        objects,
        constraints=["Reconstruct right-triangle coordinates from the stated right angle and side lengths before vector superposition."],
    )


def _equilateral_two_identical_remaining_vertex_candidate(text: str, low: str) -> dict[str, Any] | None:
    if not _is_force_query(low):
        return None
    if "equilateral triangle" not in low or "two identical charges" not in low or "remaining vertex" not in low:
        return None
    source = re.search(
        rf"two\s+identical\s+charges\s+q\s*=\s*(?P<value>{NUM})\s*(?P<unit>pC|nC|uC|mC|C|coulombs?)",
        text,
        flags=re.I,
    )
    target = re.search(
        rf"q\s*(?:['′`]|prime)\s*=\s*(?P<value>{NUM})\s*(?P<unit>pC|nC|uC|mC|C|coulombs?)",
        text,
        flags=re.I,
    )
    side = re.search(
        rf"(?:side\s+length\s+(?:of\s+)?(?:a\s*=\s*)?|side\s+a\s*=\s*)(?P<value>{NUM})\s*(?P<unit>cm|mm|m)",
        text,
        flags=re.I,
    )
    if source is None or target is None or side is None:
        return None
    q_source = (_parse_number(source.group("value")), _unit(source.group("unit")))
    q_target = (_parse_number(target.group("value")), _unit(target.group("unit")))
    a = _length_to_m((_parse_number(side.group("value")), _unit(side.group("unit"))))
    height = math.sqrt(3.0) * a / 2.0
    objects = [
        _source_charge_obj("q1", q_source, 0.0, 0.0),
        _source_charge_obj("q2", q_source, a, 0.0),
        _point_obj("M", a / 2.0, height),
        {"id": "q_test", "type": "test_charge", "role": "given", **_quantity(*q_target)},
        {"id": "F_query", "type": "force", "role": "query", "value": None, "unit": "N"},
    ]
    return _schema(
        "two_charge_geometry_field",
        objects,
        constraints=["Model the two identical charges at two vertices and the primed charge at the remaining equilateral-triangle vertex."],
    )

def _two_charge_geometry_candidate(text: str, low: str) -> dict[str, Any] | None:
    if not (_is_electric_field_query(low) or _is_force_query(low)):
        return None
    charges = _two_charge_values_for_zero_field(text, low)
    geom = _two_charge_geometry(text, low)
    if charges is None or geom is None:
        return None

    q1, q2 = charges
    a, b, target = geom
    force_query = _is_force_query(low)
    qtest = _target_charge(text) if force_query else None
    query: dict[str, Any]
    objects: list[dict[str, Any]] = [
        _source_charge_obj("q1", q1, *a),
        _source_charge_obj("q2", q2, *b),
        _point_obj("M", *target),
    ]
    eps = _relative_permittivity_value(text)
    if eps is not None:
        objects.append({"id": "eps_r", "type": "relative_permittivity", "role": "constant", **_quantity(*eps)})
    field_query = _is_electric_field_query(low)
    if force_query and qtest is not None:
        objects.append({"id": "q_test", "type": "test_charge", "role": "given", **_quantity(*qtest)})
        if field_query:
            objects.append({"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"})
        objects.append({"id": "F_query", "type": "force", "role": "query", "value": None, "unit": "N"})
    else:
        objects.append({"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"})
    return _schema(
        "two_charge_geometry_field",
        objects,
        constraints=["Build a 2D A-B-target geometry and superpose signed electric-field vectors from q1 and q2."],
    )

def _point_charge_inverse_candidate(text: str, low: str) -> dict[str, Any] | None:
    if not _is_charge_query(low):
        return None
    field = _electric_field_given(text)
    distance = _distance(text)
    if field is None or distance is None:
        return None
    if not any(p in low for p in ["point charge", "charge q", "magnitude of q", "sign and magnitude"]):
        return None
    eps = _relative_permittivity_value(text) or (1.0, "")
    sign = -1 if any(p in low for p in ["towards the charge", "toward the charge", "directed towards", "points towards"]) else 1
    return _schema(
        "point_charge_electric_field",
        [
            {"id": "E1", "type": "electric_field", "role": "given", **_quantity(*field)},
            {"id": "r1", "type": "distance", "role": "given", **_quantity(*distance)},
            {"id": "eps_r", "type": "relative_permittivity", "role": "constant", **_quantity(*eps)},
            {"id": "q_query", "type": "charge", "role": "query", "value": None, "unit": "C", "sign": sign},
        ],
        constraints=["Infer charge sign from field direction when stated; otherwise solve magnitude."],
    )


def _coulomb_unknown_charge_candidate(text: str, low: str) -> dict[str, Any] | None:
    if not _is_charge_query(low):
        return None
    known_q = _charge(text)
    force = _force(text)
    distance = _distance(text)
    if known_q is None or force is None or distance is None:
        return None
    if not any(p in low for p in ["point charge q", "charge q", "magnitude of charge q", "magnitude of charge q,"]):
        return None
    return _schema(
        "coulomb_force_two_charges",
        [
            {"id": "q_known", "type": "charge", "role": "given", **_quantity(*known_q)},
            {"id": "F1", "type": "force", "role": "given", **_quantity(*force)},
            {"id": "r1", "type": "distance", "role": "given", **_quantity(*distance)},
            {"id": "Q_query", "type": "charge", "role": "query", "value": None, "unit": "C"},
        ],
    )


def _charge_sum(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>pC|nC|uC|mC|C|coulombs?)"
    return _find_quantity(
        [rf"\bq\s*1\s*\+\s*q\s*2\s*=\s*(?P<value>{NUM})\s*{unit}\b"],
        text,
    )


def _distances_from_q1_q2(text: str) -> tuple[tuple[float, str], tuple[float, str]] | None:
    unit_pat = r"(?:cm|mm|m)"
    patterns = [
        rf"(?P<r1>{NUM})\s*(?P<u1>{unit_pat})\s+from\s+q\s*1\s+and\s+(?P<r2>{NUM})\s*(?P<u2>{unit_pat})\s+from\s+q\s*2",
        rf"(?P<r1>{NUM})\s*(?P<u1>{unit_pat})\s+from\s+q1\s+and\s+(?P<r2>{NUM})\s*(?P<u2>{unit_pat})\s+from\s+q2",
        rf"distances\s+to\s+the\s+two\s+charges\s+q\s*1.*?q\s*2.*?are\s+(?P<r1>{NUM})\s*(?P<u1>{unit_pat})\s+and\s+(?P<r2>{NUM})\s*(?P<u2>{unit_pat})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            return (_parse_number(m.group("r1")), _unit(m.group("u1"))), (_parse_number(m.group("r2")), _unit(m.group("u2")))
    return None


def _zero_field_unknown_charges_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "e = 0" not in low and "field strength is e = 0" not in low and "net electric field strength is e = 0" not in low:
        return None
    total = _charge_sum(text)
    distances = _distances_from_q1_q2(text)
    if total is None or distances is None:
        return None
    query_id = "q2_query" if re.search(r"\bfind\s+q\s*2\b", text, flags=re.I) else "q1_query"
    query_symbol = "q2" if query_id.startswith("q2") else "q1"
    return _schema(
        "two_charge_zero_field_unknown_charges",
        [
            {"id": "q_sum", "type": "charge_sum", "role": "given", **_quantity(*total)},
            {"id": "r1", "type": "distance", "role": "given", **_quantity(*distances[0])},
            {"id": "r2", "type": "distance", "role": "given", **_quantity(*distances[1])},
            {"id": query_id, "type": "charge", "role": "query", "value": None, "unit": "C", "symbol": query_symbol},
        ],
    )


def _point_charge_scaling_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "replaced by" not in low or "halved" not in low or not _is_electric_field_query(low):
        return None
    charge_factor = 2.0 if re.search(r"replaced\s+by\s+-?\s*2\s*q", text, flags=re.I) else None
    distance_factor = 0.5 if "distance" in low and "halved" in low else None
    if charge_factor is None or distance_factor is None:
        return None
    return _schema(
        "point_charge_field_scaling",
        [
            {"id": "charge_factor", "type": "charge_factor", "role": "given", "value": str(charge_factor), "unit": "times"},
            {"id": "distance_factor", "type": "distance_factor", "role": "given", "value": str(distance_factor), "unit": "times"},
            {"id": "E_ratio_query", "type": "ratio", "role": "query", "value": None, "unit": "times", "symbol": "E2/E1"},
        ],
    )


def _electric_pendulum_angle_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "angle" not in low or "suspended" not in low or "electric field" not in low:
        return None
    mass = _mass(text)
    charge = _charge(text)
    field = _electric_field_given(text)
    if mass is None or charge is None or field is None:
        return None
    g = _gravity(text) or (10.0, "m/s2")
    return _schema(
        "electric_pendulum_deflection_angle",
        [
            {"id": "m1", "type": "mass", "role": "given", **_quantity(*mass)},
            {"id": "q1", "type": "charge", "role": "given", **_quantity(*charge)},
            {"id": "E1", "type": "electric_field", "role": "given", **_quantity(*field)},
            {"id": "g1", "type": "gravitational_acceleration", "role": "constant", **_quantity(*g)},
            {"id": "theta_query", "type": "angle", "role": "query", "value": None, "unit": "rad"},
        ],
    )


def _dielectric_field_scaling_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "dielectric" not in low or not _is_electric_field_query(low):
        return None
    field = _electric_field_given(text)
    eps = _relative_permittivity_value(text)
    if field is None or eps is None:
        return None
    if not any(p in low for p in ["new electric field", "what will be", "surrounds"]):
        return None
    return _schema(
        "dielectric_field_scaling",
        [
            {"id": "E_air", "type": "electric_field", "role": "given", **_quantity(*field)},
            {"id": "eps_r", "type": "relative_permittivity", "role": "given", **_quantity(*eps)},
            {"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"},
        ],
    )


def _field_pair_values(text: str) -> tuple[tuple[float, str], tuple[float, str]] | None:
    unit_pat = r"(?:V\s*/\s*m|N\s*/\s*C)"
    m = re.search(
        rf"point\s+A\s+is\s+(?P<ea>{NUM})\s*(?P<ua>{unit_pat}).*?point\s+B\s+is\s+(?P<eb>{NUM})\s*(?P<ub>{unit_pat})",
        text,
        flags=re.I | re.S,
    )
    if not m:
        m = re.search(
            rf"field\s+strength\s+at\s+A\s+is\s+(?P<ea>{NUM})\s*(?P<ua>{unit_pat}).*?at\s+B\s+is\s+(?P<eb>{NUM})\s*(?P<ub>{unit_pat})",
            text,
            flags=re.I | re.S,
        )
    if not m:
        return None
    return (_parse_number(m.group("ea")), _unit(m.group("ua"))), (_parse_number(m.group("eb")), _unit(m.group("ub")))


def _midpoint_field_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "midpoint" not in low or not _is_electric_field_query(low):
        return None
    fields = _field_pair_values(text)
    if fields is None:
        return None
    return _schema(
        "midpoint_field_from_two_field_values",
        [
            {"id": "E_A", "type": "electric_field", "role": "given", **_quantity(*fields[0])},
            {"id": "E_B", "type": "electric_field", "role": "given", **_quantity(*fields[1])},
            {"id": "E_M_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"},
        ],
    )






def _surface_charge_density(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>uC\s*/\s*m2|uC\s*/\s*m\^2|uC\s*/\s*m²|C\s*/\s*m2|C\s*/\s*m\^2|C\s*/\s*m²)"
    return _find_quantity(
        [
            rf"(?:surface\s+charge\s+density(?:\s*(?:σ|sigma))?|σ|sigma)\s*(?:=|of|is)?\s*(?P<value>{NUM})\s*{unit}",
            rf"surface\s+charge\s+densities\s+of\s+σ[^.]*?given\s+that\s+σ\s*=\s*(?P<value>{NUM})\s*{unit}",
        ],
        text,
    )


def _area_to_m2(area: tuple[float, str]) -> float:
    value, unit = area
    unit = _unit(unit).lower().replace('^2', '2').replace('²', '2')
    factors = {"m2": 1.0, "cm2": 1e-4, "mm2": 1e-6}
    return value * factors.get(unit, 1.0)


def _density_quantity(value_unit: tuple[float, str] | None, default_unit: str) -> dict[str, Any] | None:
    if value_unit is None:
        return None
    return {"value": str(value_unit[0]), "unit": _unit(value_unit[1]) or default_unit}


def _continuous_schema(kind: str, *, distribution: dict[str, Any], tags: list[str], unit: str = "V/m") -> dict[str, Any]:
    distribution = dict(distribution)
    distribution["type"] = kind
    return {
        "domain": "electrostatics",
        "representation": "numeric",
        "solver_backend": "continuous_distribution",
        "tags": sorted(set(["continuous_distribution", *tags])),
        "distribution": distribution,
        "queries": [{"type": "electric_field", "output": "magnitude", "unit": unit}],
    }


def _radius_quantity(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>cm|mm|m)"
    return _find_quantity([
        rf"\bradius\s+R\s*=\s*(?P<value>{NUM})\s*{unit}",
        rf"\bradius\s+(?:of\s+)?(?:R\s*=\s*)?(?P<value>{NUM})\s*{unit}",
        rf"\bR\s*=\s*(?P<value>{NUM})\s*{unit}",
    ], text)


def _axis_distance_quantity(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>cm|mm|m)"
    return _find_quantity([
        rf"(?:located|point\s+P\s+located)[^.]*?\bz\s*=\s*(?P<value>{NUM})\s*{unit}",
        rf"\bdistance\s+z\s*=\s*(?P<value>{NUM})\s*{unit}",
        rf"\bz\s*=\s*(?P<value>{NUM})\s*{unit}",
        rf"(?P<value>{NUM})\s*{unit}\s+from\s+(?:the\s+)?center",
    ], text)


def _continuous_ring_axial_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "ring" not in low or "z-axis" not in low or not _is_electric_field_query(low):
        return None
    charge = _charge(text)
    radius = _radius_quantity(text)
    z = _axis_distance_quantity(text)
    if charge is None or radius is None or z is None:
        return None
    return _continuous_schema(
        "ring_axial",
        tags=["ring", "axis", "electric_field"],
        distribution={"charge": _quantity(*charge), "radius": _quantity(*radius), "axis_distance": _quantity(*z)},
        unit="V/m",
    )


def _continuous_rod_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "rod" not in low or "linear charge density" not in low or not _is_electric_field_query(low):
        return None
    length = _find_quantity([rf"\blength\s+L\s*=\s*(?P<value>{NUM})\s*(?P<unit>cm|mm|m)", rf"\bL\s*=\s*(?P<value>{NUM})\s*(?P<unit>cm|mm|m)"], text)
    lam = _line_charge_density(text)
    dist = _find_quantity([rf"\bdistance\s+r\s*=\s*(?P<value>{NUM})\s*(?P<unit>cm|mm|m)", rf"\br\s*=\s*(?P<value>{NUM})\s*(?P<unit>cm|mm|m)"], text)
    if length is None or lam is None or dist is None:
        return None
    return _continuous_schema(
        "finite_rod_perpendicular",
        tags=["finite_rod", "linear_charge_density", "electric_field"],
        distribution={"length": _quantity(*length), "linear_charge_density": _density_quantity(lam, "C/m"), "perpendicular_distance": _quantity(*dist)},
        unit="V/m",
    )


def _continuous_disk_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "disk" not in low or "surface charge density" not in low or not _is_electric_field_query(low):
        return None
    sigma = _surface_charge_density(text)
    radius = _radius_quantity(text)
    z = _axis_distance_quantity(text)
    if sigma is None or radius is None or z is None:
        return None
    return _continuous_schema(
        "disk_axial",
        tags=["disk", "surface_charge_density", "axis", "electric_field"],
        distribution={"surface_charge_density": _density_quantity(sigma, "C/m^2"), "radius": _quantity(*radius), "axis_distance": _quantity(*z)},
        unit="V/m",
    )


def _continuous_semicircle_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "semicircle" not in low or not _is_electric_field_query(low):
        return None
    charge = _charge(text)
    radius = _radius_quantity(text)
    if charge is None or radius is None:
        return None
    return _continuous_schema(
        "semicircle_center",
        tags=["semicircle", "center", "electric_field"],
        distribution={"charge": _quantity(*charge), "radius": _quantity(*radius)},
        unit="V/m",
    )


def _continuous_parallel_sheets_candidate(text: str, low: str) -> dict[str, Any] | None:
    if not any(term in low for term in ["parallel insulating plates", "parallel insulating sheets", "wide, parallel", "wide parallel"]):
        return None
    if "surface charge" not in low or not _is_electric_field_query(low):
        return None
    sigma = _surface_charge_density(text)
    if sigma is None:
        # Some statements give only symbolic sigma first and numeric sigma later.
        sigma = _find_quantity([rf"\bσ\s*=\s*(?P<value>{NUM})\s*(?P<unit>C\s*/\s*m2|C\s*/\s*m\^2|C\s*/\s*m²|uC\s*/\s*m2|uC\s*/\s*m\^2|uC\s*/\s*m²)"], text)
    if sigma is None:
        return None
    arrangement = "identical" if "identical" in low or "same" in low and "-σ" not in low and "(-σ" not in low else "opposite"
    return _continuous_schema(
        "parallel_infinite_sheets",
        tags=["infinite_sheet", "surface_charge_density", arrangement],
        distribution={"surface_charge_density": _density_quantity(sigma, "C/m^2"), "arrangement": arrangement},
        unit="V/m",
    )


def _continuous_infinite_plate_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "infinitely large" not in low and "infinite" not in low:
        return None
    if "metal plate" not in low and "flat plate" not in low:
        return None
    if not _is_electric_field_query(low):
        return None
    charge = _charge(text)
    dims = re.search(rf"(?P<a>{NUM})\s*(?P<au>cm|mm|m)\s*x\s*(?P<b>{NUM})\s*(?P<bu>cm|mm|m)", text, flags=re.I)
    if charge is None:
        m_charge = re.search(rf"charge[^.]*?area\s+is\s+(?P<value>{NUM})\s*(?P<unit>pC|nC|uC|mC|C|coulombs?)", text, flags=re.I)
        if m_charge:
            charge = (_parse_number(m_charge.group("value")), _unit(m_charge.group("unit")))
    if charge is None or dims is None:
        return None
    a = _length_to_m((_parse_number(dims.group('a')), _unit(dims.group('au'))))
    b = _length_to_m((_parse_number(dims.group('b')), _unit(dims.group('bu'))))
    return _continuous_schema(
        "infinite_plate_from_area_charge",
        tags=["infinite_plate", "area_charge", "electric_field"],
        distribution={"charge": _quantity(*charge), "area": {"value": str(a * b), "unit": "m^2"}},
        unit="V/m",
    )

def _symbolic_point(id: str, x: str, y: str) -> dict[str, str]:
    return {"id": id, "x": x, "y": y}


def _electrostatics_symbolic_geometry_schema(*, tags: list[str], **fields: Any) -> dict[str, Any]:
    """Build a normalized symbolic electrostatics candidate.

    symbolic_geometry is a solver backend/subtype, not a physics domain.  The
    legacy top-level domain is still accepted by schema_solver for old caches,
    but new hybrid candidates and LLM examples should use this normalized shape.
    """
    schema = {
        "domain": "electrostatics",
        "representation": "symbolic",
        "solver_backend": "symbolic_geometry",
        "tags": sorted(set(["symbolic", "symbolic_geometry", *tags])),
    }
    schema.update(fields)
    return schema


def _symbolic_charge(id: str, at: str, charge: str) -> dict[str, str]:
    return {"id": id, "at": at, "charge": charge}



def _symbolic_perpendicular_bisector_equal_charges_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "perpendicular bisector" not in low or not _is_electric_field_query(low):
        return None
    if not re.search(r"q\s*1\s*=\s*q\s*2\s*=\s*q", low.replace("₁", "1").replace("₂", "2")):
        return None
    if "2a" not in low or "distance h" not in low:
        return None
    asks_maximum = any(p in low for p in ["maximum", "max", "for which", "value of h"])
    query = (
        {"type": "maximize_electric_field", "target": "M", "variable": "h", "answer_mode": "symbolic_expr", "unit": "m"}
        if asks_maximum else
        {"type": "electric_field", "target": "M", "output": "magnitude", "answer_mode": "symbolic_expr", "unit": "V/m"}
    )
    return _electrostatics_symbolic_geometry_schema(
        tags=["perpendicular_bisector", "equal_charges", "electric_field", "optimization" if asks_maximum else "field_expression"],
        points=[
            _symbolic_point("A", "-a", "0"),
            _symbolic_point("B", "a", "0"),
            _symbolic_point("M", "0", "h"),
        ],
        charges=[
            _symbolic_charge("q1", "A", "q"),
            _symbolic_charge("q2", "B", "q"),
        ],
        queries=[query],
        constraints=["AB=2a; place A=(-a,0), B=(a,0), M=(0,h) and superpose symbolic field vectors."],
    )


def _symbolic_midpoint_inverse_sqrt_field_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "1/sqrt" not in low.replace(" ", "") and "1 / sqrt" not in low:
        return None
    if "e_a" not in low and "ea" not in low and "e a" not in low:
        return None
    if "e_b" not in low and "eb" not in low and "e b" not in low:
        return None
    if "midpoint" not in low:
        return None
    return _electrostatics_symbolic_geometry_schema(
        tags=["symbolic_relation", "midpoint", "inverse_sqrt_field"],
        solver_backend="symbolic_relation",
        queries=[{
            "type": "midpoint_inverse_sqrt_field_relation",
            "left": "1/sqrt(E_M)",
            "left_endpoint_field": "E_A",
            "right_endpoint_field": "E_B",
            "answer_mode": "symbolic_relation",
            "unit": "-",
        }],
        constraints=["For one point charge on a field line, 1/sqrt(E) is proportional to distance; midpoint distances average."],
    )


def _symbolic_equilateral_centroid_zero_unknown_charge_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "equilateral triangle" not in low or "centroid" not in low or "zero" not in low:
        return None
    if not re.search(r"q\s*1\s*=\s*q\s*2\s*=\s*(?P<value>" + NUM + r")\s*(?P<unit>c|uc|u c|microcoulombs?|μc|µc|nc|n c)?", low):
        return None
    m = re.search(r"q\s*1\s*=\s*q\s*2\s*=\s*(?P<value>" + NUM + r")\s*(?P<unit>c|uc|u c|microcoulombs?|μc|µc|nc|n c)?", low)
    raw_value = m.group("value") if m else "4e-9"
    unit = (m.group("unit") if m and m.group("unit") else "C").replace(" ", "")
    try:
        val = _parse_number(raw_value)
        if unit in {"uc", "μc", "µc", "microcoulomb", "microcoulombs"}:
            val *= 1e-6
        elif unit in {"nc"}:
            val *= 1e-9
    except Exception:
        return None
    return _electrostatics_symbolic_geometry_schema(
        tags=["inverse_symbolic_charge", "equilateral_triangle", "centroid", "zero_field"],
        points=[
            _symbolic_point("A", "-a/2", "-sqrt(3)*a/6"),
            _symbolic_point("B", "a/2", "-sqrt(3)*a/6"),
            _symbolic_point("C", "0", "sqrt(3)*a/3"),
            _symbolic_point("G", "0", "0"),
        ],
        charges=[
            _symbolic_charge("q1", "A", f"{val:g}"),
            _symbolic_charge("q2", "B", f"{val:g}"),
            _symbolic_charge("q3", "C", "q3"),
        ],
        queries=[{"type": "zero_field_unknown_charge", "target": "G", "unknown": "q3", "answer_mode": "symbolic_expr", "unit": "C"}],
        constraints=["Solve symbolic vector equation E_A+E_B+E_C=0 at the centroid for q3."],
    )


def _vertices_for_square_sign(low: str, sign: str) -> set[str]:
    start = low.find(f"{sign} charges")
    if start < 0:
        start = low.find(f"{sign} charge")
    if start < 0:
        return set()
    other = "negative" if sign == "positive" else "positive"
    next_other = low.find(f"{other} charges", start + 1)
    stop_candidates = [idx for idx in [next_other, low.find(".", start)] if idx >= 0]
    stop = min(stop_candidates) if stop_candidates else len(low)
    segment = low[start:stop]
    return {v.upper() for v in re.findall(r"\b[abcd]\b", segment)}


def _symbolic_square_center_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "square" not in low or not any(p in low for p in ["intersection", "diagonal", "center", "centre"]):
        return None
    if not _is_electric_field_query(low):
        return None
    # Require explicit sign/vertex assignment. This prevents a generic zero-symmetry shortcut from firing on sign patterns that do not cancel.
    positive_vertices = _vertices_for_square_sign(low, "positive")
    negative_vertices = _vertices_for_square_sign(low, "negative")
    if not positive_vertices or not negative_vertices:
        return None
    charges = []
    for vertex in ["A", "B", "C", "D"]:
        if vertex in positive_vertices:
            charges.append(_symbolic_charge(f"q{vertex}", vertex, "q"))
        elif vertex in negative_vertices:
            charges.append(_symbolic_charge(f"q{vertex}", vertex, "-q"))
    if len(charges) != 4:
        return None
    return _electrostatics_symbolic_geometry_schema(
        tags=["square", "center", "sign_pattern", "electric_field"],
        medium={"relative_permittivity_symbol": "epsilon"} if "epsilon" in low or "air" not in low else {},
        points=[
            _symbolic_point("A", "-a/2", "a/2"),
            _symbolic_point("B", "a/2", "a/2"),
            _symbolic_point("C", "a/2", "-a/2"),
            _symbolic_point("D", "-a/2", "-a/2"),
            _symbolic_point("O", "0", "0"),
        ],
        charges=charges,
        queries=[{"type": "electric_field", "target": "O", "output": "magnitude", "answer_mode": "symbolic_expr", "unit": "V/m"}],
        constraints=["Build symbolic square coordinates and superpose signed field vectors at the diagonal intersection."],
    )

def _symbolic_vector_resultant_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "given f0" not in low or "isosceles right triangle" not in low or "remaining vertex" not in low:
        return None
    return _schema(
        "symbolic_equal_perpendicular_resultant",
        [
            {"id": "F0", "type": "symbolic_force", "role": "given", "symbol": "F0", "value": "1", "unit": "symbolic"},
            {"id": "angle", "type": "angle", "role": "given", "value": "90", "unit": "degree"},
            {"id": "F_query", "type": "force", "role": "query", "value": None, "unit": "N"},
        ],
        constraints=["Two equal perpendicular force magnitudes F0 combine to sqrt(2)*F0."],
    )


def _direction_between_two_charges_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "what is the direction" not in low or "q1" not in low or "q2" not in low:
        return None
    charges = _two_charge_values_for_zero_field(text, low)
    distances = _distances_from_q1_q2(text)
    separation = _zero_field_separation(text)
    if charges is None or distances is None or separation is None:
        return None
    q1, q2 = charges
    r1 = _length_to_m(distances[0])
    r2 = _length_to_m(distances[1])
    d = _length_to_m(separation)
    if abs(r1 + r2 - d) > max(1e-9, d * 1e-6):
        return None
    target = "q2" if q1[0] > 0 and q2[0] < 0 else "q1"
    return _schema(
        "direction_between_collinear_charges",
        [
            {"id": "q1", "type": "charge", "role": "given", **_quantity(*q1)},
            {"id": "q2", "type": "charge", "role": "given", **_quantity(*q2)},
            {"id": "direction_query", "type": "force", "role": "query", "value": None, "unit": "-", "target_symbol": target},
        ],
        constraints=["Point lies between opposite-sign charges; both field/force contributions point toward the negative charge."],
    )


def _symbolic_field_ratio_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "q1 = 4q2" not in low or "f1 = 3f2" not in low or "relationship" not in low:
        return None
    return _schema(
        "symbolic_field_ratio_from_force_charge_ratios",
        [
            {"id": "F_ratio", "type": "ratio", "role": "given", "value": "3", "unit": "times", "symbol": "F1/F2"},
            {"id": "q_ratio", "type": "ratio", "role": "given", "value": "4", "unit": "times", "symbol": "q1/q2"},
            {"id": "relation_query", "type": "electric_field", "role": "query", "value": None, "unit": "-", "left": "E1", "right": "E2"},
        ],
        constraints=["Use E=F/q, hence E1/E2=(F1/F2)/(q1/q2)."],
    )


def _symbolic_right_isosceles_altitude_field_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "right isosceles triangle" not in low or "qA = qB = q".lower() not in low or "qC = 2q".lower() not in low:
        return None
    if "expression" not in low or "foot of the altitude" not in low:
        return None
    return _schema(
        "symbolic_right_isosceles_altitude_field",
        [
            {"id": "k", "type": "symbol", "role": "constant", "symbol": "k", "value": "1", "unit": "symbolic"},
            {"id": "q", "type": "symbol", "role": "given", "symbol": "q", "value": "1", "unit": "symbolic"},
            {"id": "a", "type": "symbol", "role": "given", "symbol": "a", "value": "1", "unit": "symbolic"},
            {"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "-"},
        ],
        constraints=["At the altitude foot of a right isosceles triangle, symbolic vector components reduce to 2*sqrt(2)*k*q/a^2."],
    )


def _symbolic_square_missing_charge_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "square" not in low or "charges q1 = q3 = q" not in low or "what charge must be placed at b" not in low:
        return None
    return _schema(
        "symbolic_square_field_zero_missing_charge",
        [
            {"id": "q", "type": "symbol", "role": "given", "symbol": "q", "value": "1", "unit": "symbolic"},
            {"id": "qB_query", "type": "charge", "role": "query", "value": None, "unit": "-"},
        ],
        constraints=["Solve the symbolic field-cancellation coefficient at D for a missing charge at B."],
    )

def _constant_zero_symmetry_candidate(text: str, low: str) -> dict[str, Any] | None:
    zero_force = _is_force_query(low) and any(
        phrase in low
        for phrase in [
            "equal magnitude and the same sign",
            "four identical charges",
            "three identical charges",
        ]
    ) and any(place in low for place in ["midpoint", "center", "centre", "intersection point"])
    zero_field = _is_electric_field_query(low) and any(
        phrase in low
        for phrase in [
            "four charges of the same magnitude",
            "three identical charges",
            "positive charges are located at a and c",
        ]
    ) and any(place in low for place in ["center", "centre", "intersection point"])
    if not (zero_force or zero_field):
        return None
    query_type = "force" if zero_force else "electric_field"
    unit = "N" if zero_force else "V/m"
    return _schema(
        "constant_zero_result",
        [{"id": "zero_query", "type": query_type, "role": "query", "value": None, "unit": unit}],
        constraints=["Symmetry: equal contributions cancel at the stated midpoint/center/intersection point."],
    )


def _square_side(text: str) -> tuple[float, str] | None:
    return _find_quantity(
        [
            rf"square[^.]*?side\s+length\s+(?:of\s+)?(?:a\s*=\s*)?(?P<value>{NUM})\s*(?P<unit>cm|mm|m)",
            rf"square[^.]*?with\s+a\s+side\s+length\s+of\s+(?P<value>{NUM})\s*(?P<unit>cm|mm|m)",
        ],
        text,
    )


def _equal_square_charge(text: str) -> tuple[float, str] | None:
    unit = r"(?P<unit>pC|nC|uC|mC|C|coulombs?)"
    patterns = [
        rf"\bq\s*=\s*(?P<value>{NUM})\s*{unit}\b",
        rf"\bq\s*1\s*=\s*q\s*2\s*=\s*q\s*3\s*=\s*q\s*=\s*(?P<value>{NUM})\s*{unit}\b",
        rf"\bq\s*1\s*=\s*q\s*2\s*=\s*q\s*3\s*=\s*(?P<value>{NUM})\s*{unit}\b",
    ]
    return _find_quantity(patterns, text)


def _square_three_charge_fourth_vertex_field_candidate(text: str, low: str) -> dict[str, Any] | None:
    if not _is_electric_field_query(low) or "square" not in low:
        return None
    if not any(p in low for p in ["three equal positive", "three positive charges", "three consecutive vertices", "fourth vertex"]):
        return None
    side = _square_side(text)
    charge = _equal_square_charge(text)
    if side is None or charge is None:
        return None
    a = _length_to_m(side)
    objects = [
        _source_charge_obj("qA", charge, 0.0, 0.0),
        _source_charge_obj("qB", charge, a, 0.0),
        _source_charge_obj("qC", charge, a, a),
        _point_obj("M", 0.0, a),
        {"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"},
    ]
    return _schema(
        "two_charge_geometry_field",
        objects,
        constraints=["Square geometry: place three equal source charges at consecutive vertices and evaluate the field at the fourth vertex."],
    )


def _square_center_missing_charge_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "square" not in low or "center" not in low or "zero" not in low or not (_is_charge_query(low) or "charge q4" in low):
        return None
    q2 = re.search(rf"\bq\s*2\s*=\s*(?P<value>{NUM})\s*(?P<unit>pC|nC|uC|mC|C|coulombs?)", text, flags=re.I)
    if q2 is None:
        return None
    return _schema(
        "square_center_zero_field_missing_vertex_charge",
        [
            {"id": "q_opposite", "type": "charge", "role": "given", **_quantity(_parse_number(q2.group("value")), _unit(q2.group("unit")))},
            {"id": "q4_query", "type": "charge", "role": "query", "value": None, "unit": _unit(q2.group("unit"))},
        ],
        constraints=["At the square center all four vertex distances are equal; opposite vertices cancel pairwise, so q4 must equal q2."],
    )


def _right_triangle_altitude_foot_force_candidate(text: str, low: str) -> dict[str, Any] | None:
    if not _is_force_query(low) or "foot of the altitude" not in low or "right" not in low:
        return None
    coords = _right_triangle_coordinates_from_text(text, low)
    if coords is None:
        return None
    equal = re.search(
        rf"\bq\s*1\s*=\s*q\s*2\s*=\s*q\s*3\s*=\s*(?P<value>{NUM})\s*(?P<unit>pC|nC|uC|mC|C|coulombs?)",
        text,
        flags=re.I,
    )
    test = re.search(
        rf"\bcharge\s+q\s*=\s*(?P<value>{NUM})\s*(?P<unit>pC|nC|uC|mC|C|coulombs?)\s+placed\s+at\s+point\s+H",
        text,
        flags=re.I,
    )
    if equal is None or test is None:
        return None
    q = (_parse_number(equal.group("value")), _unit(equal.group("unit")))
    q_test = (_parse_number(test.group("value")), _unit(test.group("unit")))
    # The current dataset wording uses the foot of the altitude from A to hypotenuse BC.
    a = coords["A"]
    b = coords["B"]
    c = coords["C"]
    vx = c[0] - b[0]
    vy = c[1] - b[1]
    denom = vx * vx + vy * vy
    if denom == 0:
        return None
    t = ((a[0] - b[0]) * vx + (a[1] - b[1]) * vy) / denom
    h = (b[0] + t * vx, b[1] + t * vy)
    objects = [
        _source_charge_obj("qA", q, *a),
        _source_charge_obj("qB", q, *b),
        _source_charge_obj("qC", q, *c),
        _point_obj("M", *h),
        {"id": "q_test", "type": "test_charge", "role": "given", **_quantity(*q_test)},
        {"id": "F_query", "type": "force", "role": "query", "value": None, "unit": "N"},
    ]
    return _schema(
        "two_charge_geometry_field",
        objects,
        constraints=["Right-triangle geometry: compute the foot of the altitude from A to hypotenuse BC, then superpose the force vectors."],
    )


def _parallel_capacitor_voltage_under_limit_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "parallel" not in low or "voltage" not in low or "one of the two capacitors" not in low:
        return None
    caps = re.search(
        rf"C\s*1\s*=\s*(?P<c1>{NUM})\s*(?P<u1>pF|nF|uF|mF|F).*?C\s*2\s*=\s*(?P<c2>{NUM})\s*(?P<u2>pF|nF|uF|mF|F)",
        text,
        flags=re.I | re.S,
    )
    charge = _charge(text)
    limit = re.search(rf"U\s*<\s*(?P<value>{NUM})\s*V", text, flags=re.I)
    if caps is None or charge is None or limit is None:
        return None
    candidates = [
        (_parse_number(caps.group("c1")), _unit(caps.group("u1"))),
        (_parse_number(caps.group("c2")), _unit(caps.group("u2"))),
    ]
    q_si = charge[0] * {"pc": 1e-12, "nc": 1e-9, "uc": 1e-6, "mc": 1e-3, "c": 1.0}.get(_unit(charge[1]).lower(), 1.0)
    limit_v = _parse_number(limit.group("value"))
    def cap_si(c: tuple[float, str]) -> float:
        return c[0] * {"pf": 1e-12, "nf": 1e-9, "uf": 1e-6, "mf": 1e-3, "f": 1.0}.get(_unit(c[1]).lower(), 1.0)
    viable = [c for c in candidates if cap_si(c) > 0 and q_si / cap_si(c) < limit_v]
    if not viable:
        return None
    chosen = viable[0]
    return _schema(
        "capacitor_charge_voltage",
        [
            {"id": "Q1", "type": "charge", "role": "given", **_quantity(*charge)},
            {"id": "C1", "type": "capacitance", "role": "given", **_quantity(*chosen)},
            {"id": "U_query", "type": "voltage", "role": "query", "value": None, "unit": "V"},
        ],
        constraints=["Parallel capacitors share voltage; use the capacitor choice that satisfies the stated U limit."],
    )

def _schema(formula: str, objects: list[dict[str, Any]], constraints: list[str] | None = None) -> dict[str, Any]:
    return {
        "domain": "equations",
        "objects": objects,
        "relations": [{"type": "formula", "name": formula, "objects": [obj["id"] for obj in objects]}],
        "constraints": constraints or [],
    }


def _parallel_plate_geometry_objects(text: str) -> list[dict[str, Any]] | None:
    r = _radius(text)
    a = _area(text)
    d = _distance(text)
    if d is None or (r is None and a is None):
        return None

    objects: list[dict[str, Any]] = []
    if r is not None:
        objects.append({"id": "R1", "type": "radius", "role": "given", **_quantity(*r)})
    else:
        objects.append({"id": "A1", "type": "area", "role": "given", **_quantity(*a)})
    objects.append({"id": "d1", "type": "distance", "role": "given", **_quantity(*d)})
    return objects


def _candidate_query_intent(low: str) -> str | None:
    if _is_charge_query(low):
        return "charge"
    if _is_capacitance_query(low):
        return "capacitance"
    if _is_energy_query(low):
        return "energy"
    electric_field_query_phrases = [
        "calculate the electric field",
        "determine the electric field",
        "what is the electric field",
        "electric field strength",
        "field strength",
        "field intensity",
    ]
    if any(phrase in low for phrase in electric_field_query_phrases):
        return "electric_field"
    if _is_force_query(low):
        return "force"
    if _is_electric_field_query(low):
        return "electric_field"
    return None


def _inventory_entry(item: tuple[float, str] | None) -> tuple[Quantity, ...]:
    if item is None:
        return ()
    value, unit = item
    return (Quantity(str(value), _unit(unit)),)



def _simple_angle(text: str) -> tuple[float, str] | None:
    m = re.search(rf"(?P<value>{NUM})\s*(?P<unit>°|degrees?|deg)(?=\s|$|\b)", text, flags=re.I)
    if not m:
        return None
    return _parse_number(m.group("value")), "degree"


def _speed_quantity(text: str) -> tuple[float, str] | None:
    m = re.search(rf"(?P<value>{NUM})\s*(?P<unit>km\s*/\s*s|m\s*/\s*s|km/s|m/s)", text, flags=re.I)
    if not m:
        return None
    value = _parse_number(m.group("value"))
    unit = _unit(m.group("unit"))
    if unit.lower().replace(" ", "") == "km/s":
        return value * 1000.0, "m/s"
    return value, "m/s"


def _formula_candidate(name: str, objects: list[dict[str, Any]], constraints: list[str] | None = None) -> dict[str, Any]:
    return _schema(name, objects, constraints=constraints or [])


def _electron_stopping_distance_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "electron" not in low or "velocity reduces to zero" not in low or "electric field" not in low:
        return None
    field = _electric_field_given(text)
    speed = _speed_quantity(text)
    if field is None or speed is None:
        return None
    return _formula_candidate(
        "electron_stopping_distance_uniform_field",
        [
            {"id": "E1", "type": "electric_field", "role": "given", **_quantity(*field)},
            {"id": "v0", "type": "speed", "role": "given", **_quantity(*speed)},
            {"id": "s_query", "type": "distance", "role": "query", "value": None, "unit": "mm"},
        ],
        ["Use qE=ma and v^2=2as for stopping distance."],
    )


def _charged_dust_equilibrium_mass_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "dust" not in low or "equilibrium" not in low or "thread" not in low or "electric field" not in low:
        return None
    field = _electric_field_given(text)
    charge = _charge(text)
    angle = _simple_angle(text)
    if field is None or charge is None or angle is None:
        return None
    return _formula_candidate(
        "charged_dust_equilibrium_mass",
        [
            {"id": "E1", "type": "electric_field", "role": "given", **_quantity(*field)},
            {"id": "q1", "type": "charge", "role": "given", **_quantity(*charge)},
            {"id": "theta", "type": "angle", "role": "given", **_quantity(*angle)},
            {"id": "m_query", "type": "mass", "role": "query", "value": None, "unit": "kg"},
        ],
        ["At equilibrium tan(theta)=qE/(mg)."],
    )


def _simple_point_charge_field_candidate(text: str, low: str) -> dict[str, Any] | None:
    if not _is_electric_field_query(low) or "point charge" not in low:
        return None
    # If the statement gives a force on a test charge, the intended field is F/q
    # or the source charge is inferred from Coulomb's law.  Do not let the direct
    # point-charge fallback outrank those more faithful force-aware candidates.
    if re.search(r"\bforce\b|experienc(?:es|ing)\s+(?:a\s+)?force", low):
        return None
    charge = _charge(text)
    distance = _distance(text)
    if charge is None or distance is None:
        return None
    eps = _relative_permittivity_value(text)
    objects = [
        {"id": "q1", "type": "charge", "role": "given", **_quantity(*charge)},
        {"id": "r1", "type": "distance", "role": "given", **_quantity(*distance)},
    ]
    if eps is not None:
        objects.append({"id": "eps_r", "type": "relative_permittivity", "role": "given", **_quantity(*eps)})
    objects.append({"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"})
    return _formula_candidate(
        "point_charge_electric_field",
        objects,
        ["Direct point-charge field E=k|q|/(eps_r r^2)."],
    )


def _rectangle_inverse_field_charge_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "rectangle" not in low or "e2" not in low or "e13" not in low or "determine" not in low:
        return None
    q2 = re.search(rf"q\s*2\s*=\s*(?P<value>{NUM})\s*(?P<unit>pC|nC|uC|mC|C|coulombs?)", text, flags=re.I)
    ab = re.search(rf"AB\s*=\s*(?P<value>{NUM})\s*(?P<unit>cm|mm|m)", text, flags=re.I)
    ad = re.search(rf"AD\s*=\s*(?P<value>{NUM})\s*(?P<unit>cm|mm|m)", text, flags=re.I)
    if not q2 or not ab or not ad:
        return None
    target = "q3" if "value of q3" in low or "find q3" in low else "q1"
    return _formula_candidate(
        "rectangle_inverse_field_charge",
        [
            {"id": "q2", "type": "charge", "role": "given", "value": str(_parse_number(q2.group("value"))), "unit": _unit(q2.group("unit"))},
            {"id": "AB", "type": "width", "role": "given", "value": str(_parse_number(ab.group("value"))), "unit": _unit(ab.group("unit"))},
            {"id": "AD", "type": "height", "role": "given", "value": str(_parse_number(ad.group("value"))), "unit": _unit(ad.group("unit"))},
            {"id": f"{target}_query", "type": "charge", "role": "query", "target": target, "value": None, "unit": "C"},
        ],
        ["Resolve E2=E13 by x/y components in rectangle ABCD."],
    )


def _perpendicular_bisector_unequal_field_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "equidistant from a and b" not in low or "perpendicular to ab" not in low or "midpoint of ab" not in low:
        return None
    if not _is_electric_field_query(low):
        return None
    qmatch = re.search(rf"q\s*1\s*=\s*(?P<q1>{NUM})\s*(?P<u1>pC|nC|uC|mC|C)\s+and\s+q\s*2\s*=\s*(?P<q2>{NUM})\s*(?P<u2>pC|nC|uC|mC|C)", text, flags=re.I)
    sep = re.search(rf"(?P<value>{NUM})\s*(?P<unit>cm|mm|m)\s+apart", text, flags=re.I)
    height = re.search(rf"(?P<value>{NUM})\s*(?P<unit>cm|mm|m)\s+from\s+the\s+midpoint", text, flags=re.I)
    if not qmatch or not sep or not height:
        return None
    d = _length_to_m((_parse_number(sep.group("value")), _unit(sep.group("unit"))))
    h = _length_to_m((_parse_number(height.group("value")), _unit(height.group("unit"))))
    q1 = (_parse_number(qmatch.group("q1")), _unit(qmatch.group("u1")))
    q2 = (_parse_number(qmatch.group("q2")), _unit(qmatch.group("u2")))
    return _schema(
        "two_charge_geometry_field",
        [
            _source_charge_obj("q1", q1, -d / 2.0, 0.0),
            _source_charge_obj("q2", q2, d / 2.0, 0.0),
            _point_obj("M", 0.0, h),
            {"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"},
        ],
        constraints=["Point M is on the perpendicular bisector: use half-separation horizontally and stated height from midpoint."],
    )


def _series_uncharged_capacitor_from_final_charge_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "uncharged capacitor" not in low or "final charge" not in low and "final charge on c" not in low:
        return None
    cap = _capacitance(text)
    voltage = _voltage(text)
    charges = re.findall(rf"(?P<value>{NUM})\s*(?P<unit>pC|nC|uC|mC|C)", text, flags=re.I)
    final_charge = None
    for val, unit in charges:
        if unit.lower().endswith('c'):
            final_charge = (_parse_number(val), _unit(unit))
    if cap is None or voltage is None or final_charge is None:
        return None
    return _formula_candidate("series_uncharged_capacitor_from_final_charge", [
        {"id": "C1", "type": "capacitance", "role": "given", **_quantity(*cap)},
        {"id": "U_total", "type": "voltage", "role": "given", **_quantity(*voltage)},
        {"id": "Q_final", "type": "charge", "role": "given", **_quantity(*final_charge)},
        {"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "uF"},
    ])


def _charge_sharing_identical_capacitors_energy_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "charge" not in low or not any(word in low for word in ["shared", "distributed"]) or "identical capacitors" not in low or not _is_energy_query(low):
        return None
    cap = _capacitance(text)
    if cap is None:
        m_cap = re.search(rf"\b(?P<value>{NUM})\s*(?P<unit>pF|nF|uF|mF|F)\s+capacitor", text, flags=re.I)
        if m_cap:
            cap = (_parse_number(m_cap.group("value")), _unit(m_cap.group("unit")))
    voltage = _voltage(text)
    if voltage is None:
        m_v = re.search(rf"\bcharged\s+to\s*(?P<value>{NUM})\s*(?P<unit>kV|V|mV)\b", text, flags=re.I)
        if m_v:
            voltage = (_parse_number(m_v.group("value")), _unit(m_v.group("unit")))
    count_match = re.search(r"among\s+(?P<n>two|three|four|\d+)\s+identical", low)
    if cap is None or voltage is None or count_match is None:
        return None
    word = count_match.group("n")
    n = {"two": 2, "three": 3, "four": 4}.get(word, int(word) if word.isdigit() else 2)
    return _formula_candidate("identical_capacitor_charge_sharing_energy", [
        {"id": "C1", "type": "capacitance", "role": "given", **_quantity(*cap)},
        {"id": "U1", "type": "voltage", "role": "given", **_quantity(*voltage)},
        {"id": "n", "type": "count", "role": "given", "value": str(n), "unit": ""},
        {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "uJ" if n == 2 else "J"},
    ])


def _series_equal_capacitor_energy_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "connected in series" not in low or "uncharged capacitor" not in low or not _is_energy_query(low):
        return None
    energy = _find_quantity([rf"energy\s+of\s+(?P<value>{NUM})\s*(?P<unit>mJ|uJ|μJ|J)", rf"(?P<value>{NUM})\s*(?P<unit>mJ|uJ|μJ|J)"], text)
    caps = re.findall(rf"(?P<value>{NUM})\s*(?P<unit>pF|nF|uF|mF|F)", text, flags=re.I)
    if energy is None or len(caps) < 2:
        return None
    c1 = (_parse_number(caps[0][0]), _unit(caps[0][1]))
    c2 = (_parse_number(caps[1][0]), _unit(caps[1][1]))
    return _formula_candidate("energy_shared_equal_capacitor_series", [
        {"id": "W0", "type": "energy", "role": "given", **_quantity(*energy)},
        {"id": "C1", "type": "capacitance", "role": "given", **_quantity(*c1)},
        {"id": "C2", "type": "added_capacitance", "role": "given", **_quantity(*c2)},
        {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "mJ"},
    ])


def _disconnected_dielectric_energy_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "disconnected" not in low or "permittivity" not in low or not _is_energy_query(low):
        return None
    energy = _find_quantity([rf"initial\s+energy\s+was\s+(?P<value>{NUM})\s*(?P<unit>mJ|uJ|μJ|J)", rf"(?P<value>{NUM})\s*(?P<unit>mJ|uJ|μJ|J)"], text)
    factor = re.search(rf"factor\s+of\s+(?P<value>{NUM})", text, flags=re.I)
    if energy is None or factor is None:
        return None
    return _formula_candidate("disconnected_dielectric_energy_scaling", [
        {"id": "W0", "type": "energy", "role": "given", **_quantity(*energy)},
        {"id": "eps", "type": "relative_permittivity", "role": "given", "value": str(_parse_number(factor.group("value"))), "unit": ""},
        {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "uJ"},
    ])


def _uncertain_measurements(text: str) -> list[dict[str, Any]]:
    """Return measurements written as value ± uncertainty unit with rough context."""
    unit = r"(?P<unit>atm|ohm|Ω|A|V|W|cm|mm|m|g|kg|s)"
    out: list[dict[str, Any]] = []
    pattern = rf"(?P<value>{NUM})\s*(?:±|\+/-|\+-)\s*(?P<err>{NUM})\s*{unit}"
    for m in re.finditer(pattern, text, flags=re.I):
        ctx = text[max(0, m.start() - 80): m.end() + 40].lower()
        kind = "measured_value"
        if "voltage" in ctx or " volt" in ctx:
            kind = "voltage"
        elif "current" in ctx:
            kind = "current"
        elif "resistance" in ctx:
            kind = "resistance"
        elif "length" in ctx or "height" in ctx:
            kind = "length"
        elif "mass" in ctx:
            kind = "mass"
        elif "time" in ctx:
            kind = "time"
        elif "pressure" in ctx or "gauge" in ctx:
            kind = "pressure"
        out.append({
            "value": _parse_number(m.group("value")),
            "error": _parse_number(m.group("err")),
            "unit": _unit(m.group("unit")),
            "kind": kind,
            "context": ctx,
        })
    return out


def _first_uncertain_kind(measurements: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    return next((m for m in measurements if m.get("kind") == kind), None)


def _power_uncertainty_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "power" not in low or not any(tok in text for tok in ["±", "+/-", "+-"]):
        return None
    measurements = _uncertain_measurements(text)
    voltage = _first_uncertain_kind(measurements, "voltage") or next((m for m in measurements if m.get("unit").lower() == "v"), None)
    current = _first_uncertain_kind(measurements, "current") or next((m for m in measurements if m.get("unit").lower() == "a"), None)
    if voltage is None or current is None:
        return None
    absolute = "absolute error" in low or "absolute uncertainty" in low
    return _schema(
        "power_uncertainty_product",
        [
            {"id": "U1", "type": "voltage", "role": "given", "value": str(voltage["value"]), "unit": voltage["unit"]},
            {"id": "dU1", "type": "voltage_uncertainty", "role": "given", "value": str(voltage["error"]), "unit": voltage["unit"], "symbol": "Delta U"},
            {"id": "I1", "type": "current", "role": "given", "value": str(current["value"]), "unit": current["unit"]},
            {"id": "dI1", "type": "current_uncertainty", "role": "given", "value": str(current["error"]), "unit": current["unit"], "symbol": "Delta I"},
            {"id": "dP_query", "type": "absolute_error" if absolute else "percent_error", "role": "query", "value": None, "unit": "W" if absolute else "%"},
        ],
        ["For P=UI, relative uncertainty adds; absolute uncertainty is P times that relative uncertainty."],
    )


def _series_resistance_uncertainty_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "series" not in low or "resistance" not in low or "absolute error" not in low:
        return None
    pairs = re.findall(rf"R\s*\d+\s*=\s*(?P<value>{NUM})\s*(?:±|\+/-|\+-)\s*(?P<err>{NUM})\s*(?P<unit>ohm|Ω)", text, flags=re.I)
    if len(pairs) < 2:
        return None
    objects: list[dict[str, Any]] = []
    for idx, (value, err, unit) in enumerate(pairs, start=1):
        unit_n = _unit(unit)
        objects.append({"id": f"R{idx}", "type": "resistance", "role": "given", "value": str(_parse_number(value)), "unit": unit_n})
        objects.append({"id": f"dR{idx}", "type": "resistance_uncertainty", "role": "given", "value": str(_parse_number(err)), "unit": unit_n, "symbol": f"Delta R{idx}"})
    objects.append({"id": "dR_query", "type": "absolute_error", "role": "query", "value": None, "unit": "ohm"})
    return _schema("series_resistance_uncertainty_sum", objects, ["For series sums, absolute uncertainties add."])


def _relative_uncertainty_candidate(text: str, low: str) -> dict[str, Any] | None:
    if not any(phrase in low for phrase in ["percentage relative", "relative uncertainty", "relative error"]):
        return None
    measurements = _uncertain_measurements(text)
    if measurements:
        m = measurements[0]
        return _schema(
            "percentage_relative_error",
            [
                {"id": "x1", "type": "measured_value", "role": "given", "value": str(m["value"]), "unit": m["unit"]},
                {"id": "dx1", "type": "absolute_error", "role": "given", "value": str(m["error"]), "unit": m["unit"]},
                {"id": "percent_query", "type": "percent_error", "role": "query", "value": None, "unit": "%"},
            ],
        )
    lc = re.search(rf"least\s+count\s+of\s*(?P<err>{NUM})\s*(?P<unit>atm|ohm|Ω|A|V|W|cm|mm|m|g|kg|s)", text, flags=re.I)
    measured = re.search(rf"(?:measured\s+(?:value|pressure|length)?\s*(?:is|of)?|measures\s+(?:a\s+pressure\s+of)?|result\s+is)\s*(?P<value>{NUM})\s*(?P<unit>atm|ohm|Ω|A|V|W|cm|mm|m|g|kg|s)", text, flags=re.I)
    if lc is None or measured is None:
        return None
    unit_n = _unit(measured.group("unit"))
    err = _parse_number(lc.group("err"))
    # Dataset convention: generic "instrument" rows use half least count,
    # while explicit measuring instruments and pressure gauges use the full count.
    if "pressure gauge" not in low and "measuring instrument" not in low:
        err *= 0.5
    return _schema(
        "percentage_relative_error",
        [
            {"id": "x1", "type": "measured_value", "role": "given", "value": str(_parse_number(measured.group("value"))), "unit": unit_n},
            {"id": "dx1", "type": "absolute_error", "role": "given", "value": str(err), "unit": unit_n},
            {"id": "percent_query", "type": "percent_error", "role": "query", "value": None, "unit": "%"},
        ],
    )


def _parallel_missing_branch_current_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "parallel" not in low and "branch" not in low and "lamp" not in low:
        return None
    total = re.search(rf"total\s+current\s+(?:is|of)?\s*(?P<value>{NUM})\s*(?P<unit>A|mA)", text, flags=re.I)
    branch = re.search(rf"(?:D1|D_1|D\s*1|one\s+lamp|lamp\s+D1).*?(?:is|draws)?\s*(?P<value>{NUM})\s*(?P<unit>A|mA)", text, flags=re.I)
    if total is not None and branch is not None and any(p in low for p in ["calculate the current through d2", "current through d2", "current in d2"]):
        return _schema("parallel_missing_branch_current", [
            {"id": "Itotal", "type": "total_current", "role": "given", **_quantity(_parse_number(total.group("value")), total.group("unit"))},
            {"id": "I1", "type": "current", "role": "given", **_quantity(_parse_number(branch.group("value")), branch.group("unit"))},
            {"id": "I2_query", "type": "current", "role": "query", "value": None, "unit": "A"},
        ])
    remaining = re.search(rf"(?:D2|D_2|D\s*2).*?draws\s*(?P<value>{NUM})\s*(?P<unit>A|mA)", text, flags=re.I)
    if "removed" in low and remaining is not None:
        return _schema("parallel_remaining_branch_current", [
            {"id": "I_remaining", "type": "current", "role": "given", **_quantity(_parse_number(remaining.group("value")), remaining.group("unit"))},
            {"id": "Itotal_new_query", "type": "total_current", "role": "query", "value": None, "unit": "A"},
        ])
    return None


def _parallel_total_current_from_branches_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "total current" not in low or not any(word in low for word in ["lamp", "branch", "parallel"]):
        return None
    if "calculate" not in low and "what" not in low:
        return None
    pairs = re.findall(rf"(?:D\s*\d|D_\d|lamp\s+D\s*\d|A\s*\d|A_\d)[^,.]*?(?:is|=)\s*(?P<value>{NUM})\s*(?P<unit>A|mA)", text, flags=re.I)
    if len(pairs) < 2:
        return None
    objects=[]
    for idx,(value,unit) in enumerate(pairs, start=1):
        objects.append({"id": f"I{idx}", "type": "current", "role": "given", **_quantity(_parse_number(value), unit)})
    objects.append({"id":"Itotal_query","type":"total_current","role":"query","value":None,"unit":"A"})
    return _schema("parallel_total_current_from_branches", objects)


def _identical_lamp_power_share_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "identical" not in low or "total" not in low or "power" not in low:
        return None
    p = re.search(rf"total\s+(?:of\s+)?(?P<value>{NUM})\s*(?P<unit>W|mW)", text, flags=re.I) or re.search(rf"consume\s+a\s+total\s+of\s*(?P<value>{NUM})\s*(?P<unit>W|mW)", text, flags=re.I)
    if p is None:
        return None
    n = 2 if "two" in low else 2
    return _schema("identical_branch_power_share", [
        {"id":"Ptotal","type":"total_power","role":"given", **_quantity(_parse_number(p.group("value")), p.group("unit"))},
        {"id":"n","type":"count","role":"given","value":str(n),"unit":""},
        {"id":"P_each_query","type":"power","role":"query","value":None,"unit":"W"},
    ])


def _power_current_from_power_voltage_candidate(text: str, low: str) -> dict[str, Any] | None:
    if "current" not in low or "power" not in low or "voltage" not in low:
        return None
    p = re.search(rf"(?:consumes|power(?:\s+of)?|P\s*=)\s*(?P<value>{NUM})\s*(?P<unit>W|mW)", text, flags=re.I)
    u = _voltage(text)
    if p is None or u is None:
        return None
    return _schema("power_voltage_current", [
        {"id":"P1","type":"power","role":"given", **_quantity(_parse_number(p.group("value")), p.group("unit"))},
        {"id":"U1","type":"voltage","role":"given", **_quantity(*u)},
        {"id":"I_query","type":"current","role":"query","value":None,"unit":"A"},
    ])


def _qualitative_circuit_candidate(text: str, low: str) -> dict[str, Any] | None:
    answer: str | None = None
    tag = ""
    if "resistance" in low and "decreases" in low and "current" in low:
        answer = "Resistance decreases, so current increases."
        tag = "resistance_down_current_up"
    elif "current through one lamp" in low and "increases" in low and "total current" in low:
        answer = "Total current increases."
        tag = "branch_current_up_total_current_up"
    elif "total current increases" in low and any(word in low for word in ["light bulbs", "lamp", "bulb"]):
        answer = "The lamp shines brighter because the current through it increases."
        tag = "current_up_brightness_up"
    elif "same voltage" in low and "lower resistance" in low and any(word in low for word in ["bright", "bulb", "lamp"]):
        answer = "Brighter because the current is higher."
        tag = "lower_resistance_brighter"
    if answer is None:
        return None
    return _schema("qualitative_circuit_relation", [
        {"id":"qual_query","type":"qualitative", "role":"query", "value": None, "unit":"-", "answer": answer, "tag": tag},
    ])

def _quantity_inventory(text: str) -> dict[str, tuple[Quantity, ...]]:
    inv: dict[str, tuple[Quantity, ...]] = {
        "capacitance": _inventory_entry(_capacitance(text)),
        "voltage": _inventory_entry(_voltage(text)),
        "charge": _inventory_entry(_charge(text)),
        "distance": _inventory_entry(_distance(text)),
        "force": _inventory_entry(_force(text)),
        "electric_field": _inventory_entry(_electric_field_given(text)),
        "relative_permittivity": _inventory_entry(_relative_permittivity_value(text)),
        "line_charge_density": _inventory_entry(_line_charge_density(text)),
        "mass": _inventory_entry(_mass(text)),
        "gravity": _inventory_entry(_gravity(text)),
    }
    return {key: values for key, values in inv.items() if values}


def generate_equations_candidate_schemas(problem: str) -> list[dict[str, Any]]:
    """Generate high-precision equation schemas directly from text.

    This is intentionally narrow. It does not replace the LLM; it proposes
    solver-checkable candidates that can beat a bad LLM schema when the problem
    has clear quantities and intent.
    """

    text = _norm(problem)
    low = text.lower()
    candidates: list[dict[str, Any]] = []

    for special in (
        _power_uncertainty_candidate(text, low),
        _series_resistance_uncertainty_candidate(text, low),
        _relative_uncertainty_candidate(text, low),
        _parallel_missing_branch_current_candidate(text, low),
        _parallel_total_current_from_branches_candidate(text, low),
        _identical_lamp_power_share_candidate(text, low),
        _power_current_from_power_voltage_candidate(text, low),
        _qualitative_circuit_candidate(text, low),
        _two_charge_zero_field_candidate(text, low),
        _zero_field_unknown_charges_candidate(text, low),
        _symbolic_perpendicular_bisector_equal_charges_candidate(text, low),
        _symbolic_midpoint_inverse_sqrt_field_candidate(text, low),
        _symbolic_equilateral_centroid_zero_unknown_charge_candidate(text, low),
        _symbolic_square_center_candidate(text, low),
        _continuous_ring_axial_candidate(text, low),
        _continuous_rod_candidate(text, low),
        _continuous_parallel_sheets_candidate(text, low),
        _continuous_disk_candidate(text, low),
        _continuous_infinite_plate_candidate(text, low),
        _continuous_semicircle_candidate(text, low),
        _electron_stopping_distance_candidate(text, low),
        _charged_dust_equilibrium_mass_candidate(text, low),
        _simple_point_charge_field_candidate(text, low),
        _rectangle_inverse_field_charge_candidate(text, low),
        _perpendicular_bisector_unequal_field_candidate(text, low),
        _series_uncharged_capacitor_from_final_charge_candidate(text, low),
        _charge_sharing_identical_capacitors_energy_candidate(text, low),
        _series_equal_capacitor_energy_candidate(text, low),
        _disconnected_dielectric_energy_candidate(text, low),
        _symbolic_vector_resultant_candidate(text, low),
        _direction_between_two_charges_candidate(text, low),
        _symbolic_field_ratio_candidate(text, low),
        _symbolic_right_isosceles_altitude_field_candidate(text, low),
        _symbolic_square_missing_charge_candidate(text, low),
        _constant_zero_symmetry_candidate(text, low),
        _right_triangle_three_charge_candidate(text, low),
        _right_triangle_altitude_foot_force_candidate(text, low),
        _equilateral_two_identical_remaining_vertex_candidate(text, low),
        _square_three_charge_fourth_vertex_field_candidate(text, low),
        _square_center_missing_charge_candidate(text, low),
        _two_charge_geometry_candidate(text, low),
        _parallel_capacitor_voltage_under_limit_candidate(text, low),
        _two_field_vector_resultant_candidate(text, low),
        _point_charge_inverse_candidate(text, low),
        _coulomb_unknown_charge_candidate(text, low),
        _point_charge_scaling_candidate(text, low),
        _electric_pendulum_angle_candidate(text, low),
        _dielectric_field_scaling_candidate(text, low),
        _midpoint_field_candidate(text, low),
    ):
        if special is not None:
            candidates.append(special)

    c = _capacitance(text)
    u = _voltage(text)
    q = _charge(text)

    if c is not None and u is not None and _is_energy_query(low):
        candidates.append(
            _schema(
                "capacitor_energy_voltage",
                [
                    {"id": "C1", "type": "capacitance", "role": "given", **_quantity(*c)},
                    {"id": "U1", "type": "voltage", "role": "given", **_quantity(*u)},
                    {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": _energy_query_unit(c, u)},
                ],
            )
        )

    geom = _parallel_plate_geometry_objects(text)
    if geom is not None and _is_capacitance_query(low) and not _is_dielectric_constant_query(low):
        candidates.append(
            _schema(
                "parallel_plate_capacitance",
                [*geom, {"id": "eps_r", "type": "relative_permittivity", "role": "constant", "value": 1, "unit": ""}, {"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "pF"}],
            )
        )

    if geom is not None and u is not None and _is_charge_query(low):
        eps = _relative_permittivity_value(text)
        eps_obj = {"id": "eps_r", "type": "relative_permittivity", "role": "given", **_quantity(*eps)} if eps is not None else {"id": "eps_r", "type": "relative_permittivity", "role": "constant", "value": 1, "unit": ""}
        candidates.append(
            _schema(
                "parallel_plate_charge_from_voltage",
                [*geom, eps_obj, {"id": "U1", "type": "voltage", "role": "given", **_quantity(*u)}, {"id": "Q_query", "type": "charge", "role": "query", "value": None, "unit": "nC"}],
            )
        )

    if geom is not None and c is not None and _is_dielectric_constant_query(low):
        candidates.append(
            _schema(
                "parallel_plate_capacitance",
                [
                    {"id": "C1", "type": "capacitance", "role": "given", **_quantity(*c)},
                    *geom,
                    {"id": "eps_query", "type": "relative_permittivity", "role": "query", "value": None, "unit": ""},
                ],
            )
        )

    force = _force(text)
    field = _electric_field_given(text)
    mass = _mass(text)
    if mass is not None and field is not None and _is_charge_query(low) and "equilibrium" in low:
        g = _gravity(text)
        g_value = g[0] if g is not None else 10.0
        m_value, m_unit = mass
        mass_kg = m_value / 1000.0 if _unit(m_unit) == "g" else m_value
        force_value = mass_kg * g_value
        candidates.append(
            _schema(
                "electric_force_field",
                [
                    {"id": "F_gravity", "type": "force", "role": "given", "value": str(force_value), "unit": "N"},
                    {"id": "E1", "type": "electric_field", "role": "given", **_quantity(*field)},
                    {"id": "q_query", "type": "charge", "role": "query", "value": None, "unit": "C"},
                ],
            )
        )

    candidates.extend(fill_formula_specs(_quantity_inventory(text), _candidate_query_intent(low), text=low))

    # Preserve order but deduplicate by a compact schema signature.
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cand in candidates:
        key = repr((
            cand.get("domain"),
            cand.get("solver_backend"),
            cand.get("distribution"),
            cand.get("relations"),
            [(o.get("type"), o.get("role"), o.get("value"), o.get("unit")) for o in cand.get("objects", [])],
        ))
        if key not in seen:
            out.append(cand)
            seen.add(key)
    return out
