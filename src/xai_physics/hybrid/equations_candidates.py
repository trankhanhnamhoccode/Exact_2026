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
            rf"\belectric\s+field(?:\s+strength)?(?:\s+E)?(?:\s+of|\s+is|\s*=|\s+with\s+(?:a\s+)?magnitude\s+of)?\s*(?P<value>{NUM})\s*{unit}\b",
            rf"\bfield\s+strength(?:\s+E)?(?:\s+of|\s+is|\s*=)?\s*(?P<value>{NUM})\s*{unit}\b",
        ],
        text,
    )


def _relative_permittivity_value(text: str) -> tuple[float, str] | None:
    return _find_quantity(
        [
            rf"\b(?:dielectric\s+constant|relative\s+permittivity|epsilon|eps|ε)\s*(?:=|is|of)?\s*(?P<value>{NUM})\s*(?P<unit>)",
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

    zero_field_candidate = _two_charge_zero_field_candidate(text, low)
    if zero_field_candidate is not None:
        candidates.append(zero_field_candidate)

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
