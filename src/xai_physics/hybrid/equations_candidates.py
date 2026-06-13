from __future__ import annotations

import re
from typing import Any

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
    supers = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    text = text.translate(supers)
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
            rf"\br\s*=\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\b(?P<value>{NUM})\s*{unit}\s+away\s+from\b",
            rf"\bat\s+(?:a\s+)?point\s+(?P<value>{NUM})\s*{unit}\s+away\b",
            rf"\b(?P<value>{NUM})\s*{unit}\s+from\s+(?:the\s+)?(?:sphere|charge|wire)\b",
            rf"\bMO\s*=\s*(?P<value>{NUM})\s*{unit}\b",
        ],
        text,
    )


def _is_charge_query(low: str) -> bool:
    return any(p in low for p in ["calculate the charge", "what is the charge", "find the charge", "charge stored", "charge on", "charge accumulated", "maximum charge"])


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


def generate_equations_candidate_schemas(problem: str) -> list[dict[str, Any]]:
    """Generate high-precision equation schemas directly from text.

    This is intentionally narrow. It does not replace the LLM; it proposes
    solver-checkable candidates that can beat a bad LLM schema when the problem
    has clear quantities and intent.
    """

    text = _norm(problem)
    low = text.lower()
    candidates: list[dict[str, Any]] = []

    c = _capacitance(text)
    u = _voltage(text)
    q = _charge(text)

    if c is not None and u is not None and _is_charge_query(low):
        candidates.append(
            _schema(
                "capacitor_charge_voltage",
                [
                    {"id": "C1", "type": "capacitance", "role": "given", **_quantity(*c)},
                    {"id": "U1", "type": "voltage", "role": "given", **_quantity(*u)},
                    {"id": "Q_query", "type": "charge", "role": "query", "value": None, "unit": "nC"},
                ],
            )
        )

    if q is not None and u is not None and _is_capacitance_query(low):
        candidates.append(
            _schema(
                "capacitor_charge_voltage",
                [
                    {"id": "Q1", "type": "charge", "role": "given", **_quantity(*q)},
                    {"id": "U1", "type": "voltage", "role": "given", **_quantity(*u)},
                    {"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "uF"},
                ],
            )
        )

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

    if q is not None and _distance(text) is not None and _is_electric_field_query(low) and any(p in low for p in ["point charge", "small sphere", "electric charge", "charge q", "charge of"]):
        r = _distance(text)
        candidates.append(
            _schema(
                "point_charge_electric_field",
                [
                    {"id": "q1", "type": "charge", "role": "given", **_quantity(*q)},
                    {"id": "r1", "type": "distance", "role": "given", **_quantity(*r)},
                    {"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"},
                ],
            )
        )

    line_charge = _line_charge_density(text)
    if line_charge is not None and _distance(text) is not None and _is_electric_field_query(low) and any(p in low for p in ["wire", "linear charge density", "λ", "lambda"]):
        r = _distance(text)
        candidates.append(
            _schema(
                "infinite_wire_electric_field",
                [
                    {"id": "lambda1", "type": "line_charge_density", "role": "given", **_quantity(*line_charge)},
                    {"id": "r1", "type": "distance", "role": "given", **_quantity(*r)},
                    {"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"},
                ],
            )
        )

    mass = _mass(text)
    if mass is not None and q is not None and _is_electric_field_query(low) and "equilibrium" in low:
        objects = [
            {"id": "m1", "type": "mass", "role": "given", **_quantity(*mass)},
            {"id": "q1", "type": "charge", "role": "given", **_quantity(*q)},
        ]
        g = _gravity(text)
        if g is not None:
            objects.append({"id": "g1", "type": "gravitational_acceleration", "role": "given", **_quantity(*g)})
        objects.append({"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "V/m"})
        candidates.append(_schema("equilibrium_electric_field", objects))

    # Preserve order but deduplicate by a compact schema signature.
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cand in candidates:
        key = repr((cand.get("relations"), [(o.get("type"), o.get("role"), o.get("value"), o.get("unit")) for o in cand.get("objects", [])]))
        if key not in seen:
            out.append(cand)
            seen.add(key)
    return out
