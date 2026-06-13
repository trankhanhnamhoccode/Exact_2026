from __future__ import annotations

import re
from typing import Any

from xai_physics.hybrid.formula_filler import Quantity, fill_formula_specs

NUM = r"[-+]?(?:(?:\d+(?:\.\d+)?|\.\d+)(?:\s*(?:×|x|\*)\s*10\s*(?:\^|\*\*)?\s*[-+]?\d+|[eE][-+]?\d+)?|10\s*(?:\^|\*\*)\s*[-+]?\d+|10\s*[-+]\s*\d+)"


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
    return text


def _parse_number(raw: str) -> float:
    cleaned = raw.strip().replace(",", "")
    cleaned = re.sub(r"^([-+]?)10\s*(?:\^|\*\*)\s*([-+]?\d+)$", r"\g<1>1e\2", cleaned)
    cleaned = re.sub(r"^([-+]?)10\s*([-+]\s*\d+)$", lambda m: f"{m.group(1)}1e{m.group(2).replace(' ', '')}", cleaned)
    cleaned = re.sub(r"\s*(?:×|x|\*)\s*10\s*(?:\^|\*\*)?\s*", "e", cleaned)
    cleaned = cleaned.replace(" ", "")
    return float(cleaned)


def _unit(raw: str) -> str:
    unit = raw.replace("µ", "u").replace("μ", "u").replace("²", "2").replace("^2", "2")
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
            rf"\b(?:points\s+)?A\s+and\s+B\s+are\s*(?P<value>{NUM})\s*{unit}\s+apart\b",
            rf"\b(?:points\s+)?A\s+and\s+B\b[^.]*?\b(?P<value>{NUM})\s*{unit}\s+apart\b",
            rf"\bAB\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bq\s*2\b[^.]*?\blocated\s+(?P<value>{NUM})\s*{unit}\s+from\s+the\s+origin\b",
        ],
        text,
    )


def _is_zero_field_distance_query(low: str) -> bool:
    has_zero_field = "zero" in low and any(p in low for p in ["electric field", "field strength", "resultant electric field", "net electric field"])
    asks_location = any(p in low for p in ["where", "find point", "calculate its distance", "calculate the distance", "coordinate"])
    return has_zero_field and asks_location


def _zero_field_reference(text: str, low: str) -> str:
    if re.search(r"\bBM\b", text, flags=re.I) or "distance from b" in low:
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
        _two_charge_zero_field_candidate(text, low),
        _zero_field_unknown_charges_candidate(text, low),
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
        candidates.append(
            _schema(
                "parallel_plate_charge_from_voltage",
                [*geom, {"id": "U1", "type": "voltage", "role": "given", **_quantity(*u)}, {"id": "Q_query", "type": "charge", "role": "query", "value": None, "unit": "nC"}],
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
        key = repr((cand.get("relations"), [(o.get("type"), o.get("role"), o.get("value"), o.get("unit")) for o in cand.get("objects", [])]))
        if key not in seen:
            out.append(cand)
            seen.add(key)
    return out
