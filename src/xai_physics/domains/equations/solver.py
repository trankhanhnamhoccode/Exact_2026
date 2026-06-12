from __future__ import annotations

import math
from typing import Any

from xai_physics.core.result import SolveResult
from xai_physics.core.units import UNIT_TO_SI, normalize_unit, to_si


EPS0 = 8.854187817e-12


def _new_result(status: str = "ok") -> SolveResult:
    return SolveResult(status=status, domain="equations")


def _fail(message: str, formula: str | None = None) -> SolveResult:
    result = _new_result("solve_failed")
    result.error = message
    result.add_step(
        "Equation solve failed",
        message if formula is None else f"{formula}: {message}",
    )
    return result


def _fmt(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.12g}"





def _unit_factor(unit: str | None) -> float:
    if unit is None:
        return 1.0

    normalized = normalize_unit(str(unit))
    if normalized in (None, "", "-", "1", "unitless", "none"):
        return 1.0

    aliases = {
        # dimensionless / count
        "times": 1.0,
        "x": 1.0,
        "%": 1.0,
        "turn": 1.0,
        "turns": 1.0,
        "turns/m": 1.0,

        # capacitance
        "F": 1.0,
        "mF": 1e-3,
        "uF": 1e-6,
        "?F": 1e-6,
        "?F": 1e-6,
        "?F": 1e-6,
        "nF": 1e-9,
        "pF": 1e-12,

        # charge
        "C": 1.0,
        "mC": 1e-3,
        "uC": 1e-6,
        "?C": 1e-6,
        "?C": 1e-6,
        "?C": 1e-6,
        "nC": 1e-9,
        "pC": 1e-12,

        # voltage/current/resistance/power/energy/frequency/time
        "V": 1.0,
        "kV": 1e3,
        "mV": 1e-3,
        "A": 1.0,
        "mA": 1e-3,
        "?": 1.0,
        "?": 1.0,
        "ohm": 1.0,
        "ohms": 1.0,
        "W": 1.0,
        "mW": 1e-3,
        "J": 1.0,
        "mJ": 1e-3,
        "uJ": 1e-6,
        "?J": 1e-6,
        "?J": 1e-6,
        "?J": 1e-6,
        "nJ": 1e-9,
        "Hz": 1.0,
        "kHz": 1e3,
        "rad/s": 1.0,
        "s": 1.0,
        "ms": 1e-3,

        # inductance / magnetism
        "H": 1.0,
        "mH": 1e-3,
        "uH": 1e-6,
        "?H": 1e-6,
        "?H": 1e-6,
        "?H": 1e-6,
        "T": 1.0,
        "mT": 1e-3,
        "Wb": 1.0,
        "mWb": 1e-3,
        "uWb": 1e-6,
        "?Wb": 1e-6,
        "?Wb": 1e-6,
        "?Wb": 1e-6,

        # geometry / field
        "m": 1.0,
        "cm": 1e-2,
        "mm": 1e-3,
        "m2": 1.0,
        "m^2": 1.0,
        "m?": 1.0,
        "cm2": 1e-4,
        "cm^2": 1e-4,
        "cm?": 1e-4,
        "mm2": 1e-6,
        "mm^2": 1e-6,
        "mm?": 1e-6,
        "m3": 1.0,
        "m^3": 1.0,
        "m?": 1.0,
        "cm3": 1e-6,
        "cm^3": 1e-6,
        "cm?": 1e-6,
        "J/m3": 1.0,
        "J/m^3": 1.0,
        "J/m?": 1.0,
        "V/m": 1.0,
        "N/C": 1.0,
    }
    if normalized in aliases:
        return aliases[normalized]

    if normalized not in UNIT_TO_SI:
        raise ValueError(f"Unsupported unit: {unit}")

    entry = UNIT_TO_SI[normalized]
    if isinstance(entry, tuple):
        return float(entry[0])
    return float(entry)

def _raw_value(quantity: Any) -> Any:
    if isinstance(quantity, dict):
        return quantity.get("value")
    return quantity


def _raw_unit(quantity: Any, default_unit: str = "") -> str:
    if isinstance(quantity, dict):
        unit = quantity.get("unit")
        if unit not in (None, ""):
            return str(unit)
    return default_unit



def _to_si_quantity(quantity: Any, default_unit: str = "") -> float:
    value = _raw_value(quantity)
    unit = _raw_unit(quantity, default_unit)

    if value is None:
        raise ValueError("Missing numeric value")

    # schemas t? notebook/dataset th??ng l?u number d??i d?ng string: "100", "3e5", "1.2e-5"
    s = str(value).replace(",", "").strip()
    try:
        numeric = float(s)
    except ValueError as exc:
        raise ValueError(f"Could not parse numeric value: {value!r}") from exc

    return numeric * _unit_factor(unit)

def _from_si(value_si: float, unit: str | None) -> float:
    return value_si / _unit_factor(unit)


def _objects(schema: dict[str, Any]) -> list[dict[str, Any]]:
    objects = schema.get("objects", [])
    if not isinstance(objects, list):
        return []
    return [obj for obj in objects if isinstance(obj, dict)]


def _relations(schema: dict[str, Any]) -> list[dict[str, Any]]:
    relations = schema.get("relations", [])
    if isinstance(relations, dict):
        return [relations]
    if not isinstance(relations, list):
        return []
    return [rel for rel in relations if isinstance(rel, dict)]


def _first_formula_relation(schema: dict[str, Any]) -> dict[str, Any]:
    for rel in _relations(schema):
        if rel.get("type") == "formula" or rel.get("name") or rel.get("formula"):
            return rel
    return {}


def _formula_id(schema: dict[str, Any]) -> str:
    for source in (schema, _first_formula_relation(schema)):
        for key in ("formula", "formula_id", "name", "type", "relation"):
            value = source.get(key)
            if isinstance(value, str) and value.strip() and value != "formula":
                return value.strip()
    return ""


def _relation_objects(schema: dict[str, Any], relation: dict[str, Any]) -> list[dict[str, Any]]:
    id_map = {obj.get("id"): obj for obj in _objects(schema)}
    out: list[dict[str, Any]] = []
    for obj_id in relation.get("objects", []):
        obj = id_map.get(obj_id)
        if obj is not None:
            out.append(obj)
    return out


def _all_relevant_objects(schema: dict[str, Any]) -> list[dict[str, Any]]:
    rel = _first_formula_relation(schema)
    rel_objs = _relation_objects(schema, rel)
    return rel_objs if rel_objs else _objects(schema)


def _find_obj(
    schema: dict[str, Any],
    obj_type: str | tuple[str, ...],
    *,
    role: str | None = None,
) -> dict[str, Any] | None:
    types = (obj_type,) if isinstance(obj_type, str) else obj_type
    for obj in _all_relevant_objects(schema):
        if obj.get("type") in types and (role is None or obj.get("role") == role):
            return obj
    for obj in _objects(schema):
        if obj.get("type") in types and (role is None or obj.get("role") == role):
            return obj
    return None


def _query_obj(schema: dict[str, Any], obj_type: str | tuple[str, ...] | None = None) -> dict[str, Any] | None:
    if obj_type is None:
        for obj in _all_relevant_objects(schema):
            if obj.get("role") == "query":
                return obj
        return None
    return _find_obj(schema, obj_type, role="query")


def _given_obj(schema: dict[str, Any], obj_type: str | tuple[str, ...]) -> dict[str, Any] | None:
    obj = _find_obj(schema, obj_type, role="given")
    if obj is not None:
        return obj
    return _find_obj(schema, obj_type, role="constant")


def _required_si(schema: dict[str, Any], obj_type: str | tuple[str, ...], unit: str) -> float:
    obj = _given_obj(schema, obj_type)
    if obj is None:
        raise ValueError(f"Missing given object of type {obj_type}")
    return _to_si_quantity(obj, unit)


def _relative_permittivity(schema: dict[str, Any]) -> float:
    obj = _given_obj(schema, ("relative_permittivity", "dielectric_constant"))
    if obj is None:
        return 1.0
    return float(_raw_value(obj))


def _area_si(schema: dict[str, Any]) -> float:
    area = _given_obj(schema, "area")
    if area is not None:
        return _to_si_quantity(area, "m2")

    radius = _given_obj(schema, "radius")
    if radius is not None:
        r = _to_si_quantity(radius, "m")
        return math.pi * r * r

    raise ValueError("Missing area or radius")


def _answer(
    value_si: float,
    query: dict[str, Any] | None,
    quantity_type: str,
    default_unit: str,
) -> str:
    unit = default_unit
    if query is not None and query.get("unit") not in (None, ""):
        unit = str(query["unit"])

    if unit in ("", "-", None):
        return _fmt(value_si)

    return f"{_fmt(_from_si(value_si, unit))} {unit}"


def _solve_capacitor_charge_voltage(schema: dict[str, Any], formula: str) -> SolveResult:
    q_query = _query_obj(schema, "charge")
    c_query = _query_obj(schema, "capacitance")
    u_query = _query_obj(schema, "voltage")

    try:
        if q_query is not None:
            c = _required_si(schema, "capacitance", "F")
            u = _required_si(schema, "voltage", "V")
            value = c * u
            query = q_query
            quantity_type = "charge"
            default_unit = "C"
        elif c_query is not None:
            q = _required_si(schema, "charge", "C")
            u = _required_si(schema, "voltage", "V")
            if u == 0:
                return _fail("Voltage must be non-zero when solving capacitance.", formula)
            value = q / u
            query = c_query
            quantity_type = "capacitance"
            default_unit = "F"
        elif u_query is not None:
            q = _required_si(schema, "charge", "C")
            c = _required_si(schema, "capacitance", "F")
            if c == 0:
                return _fail("Capacitance must be non-zero when solving voltage.", formula)
            value = q / c
            query = u_query
            quantity_type = "voltage"
            default_unit = "V"
        else:
            return _fail("No supported query object for Q = C*U.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use capacitor charge-voltage relation Q = C*U.")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_capacitor_energy_voltage(schema: dict[str, Any], formula: str) -> SolveResult:
    w_query = _query_obj(schema, "energy")
    c_query = _query_obj(schema, "capacitance")
    u_query = _query_obj(schema, "voltage")

    try:
        if w_query is not None:
            c = _required_si(schema, "capacitance", "F")
            u = _required_si(schema, "voltage", "V")
            value = 0.5 * c * u * u
            query = w_query
            quantity_type = "energy"
            default_unit = "J"
        elif c_query is not None:
            w = _required_si(schema, "energy", "J")
            u = _required_si(schema, "voltage", "V")
            if u == 0:
                return _fail("Voltage must be non-zero when solving capacitance.", formula)
            value = 2 * w / (u * u)
            query = c_query
            quantity_type = "capacitance"
            default_unit = "F"
        elif u_query is not None:
            w = _required_si(schema, "energy", "J")
            c = _required_si(schema, "capacitance", "F")
            if c == 0:
                return _fail("Capacitance must be non-zero when solving voltage.", formula)
            value = math.sqrt(2 * w / c)
            query = u_query
            quantity_type = "voltage"
            default_unit = "V"
        else:
            return _fail("No supported query object for W = 0.5*C*U^2.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use capacitor energy formula W = 1/2*C*U^2.")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_parallel_plate_capacitance(schema: dict[str, Any], formula: str) -> SolveResult:
    c_query = _query_obj(schema, "capacitance")
    eps_query = _query_obj(schema, ("relative_permittivity", "dielectric_constant"))

    try:
        area = _area_si(schema)
        d = _required_si(schema, ("distance", "length"), "m")

        if d == 0:
            return _fail("Plate separation must be non-zero.", formula)

        if c_query is not None:
            eps_r = _relative_permittivity(schema)
            value = EPS0 * eps_r * area / d
            query = c_query
            quantity_type = "capacitance"
            default_unit = "pF"
        elif eps_query is not None:
            c = _required_si(schema, "capacitance", "F")
            value = c * d / (EPS0 * area)
            query = eps_query
            quantity_type = "relative_permittivity"
            default_unit = "-"
        else:
            return _fail("No supported query object for C = eps0*eps_r*A/d.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use parallel-plate capacitance C = eps0*eps_r*A/d.")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_parallel_plate_charge_from_voltage(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "charge")
    if query is None:
        return _fail("No charge query object for Q = eps0*eps_r*A*U/d.", formula)

    try:
        area = _area_si(schema)
        d = _required_si(schema, ("distance", "length"), "m")
        u = _required_si(schema, "voltage", "V")
        eps_r = _relative_permittivity(schema)

        if d == 0:
            return _fail("Plate separation must be non-zero.", formula)

        value = EPS0 * eps_r * area * u / d
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use Q = eps0*eps_r*A*U/d.")
    result.answer = _answer(value, query, "charge", "nC")
    return result


def _solve_parallel_plate_charge_from_field(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "charge")
    if query is None:
        return _fail("No charge query object for Q = eps0*eps_r*A*E.", formula)

    try:
        area = _area_si(schema)
        e = _required_si(schema, "electric_field", "V/m")
        eps_r = _relative_permittivity(schema)
        value = EPS0 * eps_r * area * e
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use Q = eps0*eps_r*A*E.")
    result.answer = _answer(value, query, "charge", "uC")
    return result


def _solve_parallel_plate_field(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "electric_field")
    if query is None:
        return _fail("No electric field query object for E = U/d.", formula)

    try:
        u = _required_si(schema, "voltage", "V")
        d = _required_si(schema, ("distance", "length"), "m")
        if d == 0:
            return _fail("Plate separation must be non-zero.", formula)
        value = u / d
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use parallel-plate field E = U/d.")
    result.answer = _answer(value, query, "electric_field", "V/m")
    return result


def _solve_capacitor_energy_density(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "energy_density")
    if query is None:
        return _fail("No energy density query object for u = 0.5*eps0*eps_r*E^2.", formula)

    try:
        e_obj = _given_obj(schema, "electric_field")
        if e_obj is not None:
            e = _to_si_quantity(e_obj, "V/m")
        else:
            u = _required_si(schema, "voltage", "V")
            d = _required_si(schema, ("distance", "length"), "m")
            if d == 0:
                return _fail("Plate separation must be non-zero.", formula)
            e = u / d

        eps_r = _relative_permittivity(schema)
        value = 0.5 * EPS0 * eps_r * e * e
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use electric field energy density u = 1/2*eps0*eps_r*E^2.")
    result.answer = _answer(value, query, "energy_density", "J/m3")
    return result



def _id_symbol_text(obj: dict[str, Any]) -> str:
    return " ".join(
        str(obj.get(k, ""))
        for k in ("id", "symbol", "type")
    ).lower()


def _objects_by_type(
    schema: dict[str, Any],
    obj_type: str | tuple[str, ...],
    *,
    role: str | None = None,
) -> list[dict[str, Any]]:
    types = (obj_type,) if isinstance(obj_type, str) else obj_type
    out: list[dict[str, Any]] = []
    seen: set[int] = set()

    for source in (_all_relevant_objects(schema), _objects(schema)):
        for obj in source:
            if id(obj) in seen:
                continue
            if obj.get("type") not in types:
                continue
            if role is not None and obj.get("role") != role:
                continue
            out.append(obj)
            seen.add(id(obj))

    return out


def _find_obj_by_alias(
    schema: dict[str, Any],
    aliases: tuple[str, ...],
    *,
    obj_type: str | tuple[str, ...] | None = None,
    role: str | None = None,
) -> dict[str, Any] | None:
    if obj_type is None:
        candidates = _all_relevant_objects(schema) + _objects(schema)
    else:
        candidates = _objects_by_type(schema, obj_type, role=role)

    aliases_l = tuple(a.lower() for a in aliases)

    for obj in candidates:
        if role is not None and obj.get("role") != role:
            continue
        blob = _id_symbol_text(obj)
        if any(a in blob for a in aliases_l):
            return obj

    return None


def _dimensionless_value(obj: dict[str, Any]) -> float:
    unit = _raw_unit(obj, "times")
    if normalize_unit(unit) == "%":
        return _to_si_quantity(obj, "%") / 100.0
    return _to_si_quantity(obj, "times")


def _answer_ratio(value: float, query: dict[str, Any] | None) -> str:
    unit = "times"
    if query is not None and query.get("unit") not in (None, ""):
        unit = str(query["unit"])

    if normalize_unit(unit) == "%":
        return f"{_fmt(value * 100.0)} %"

    return f"{_fmt(value)} times"


def _capacitance_ratio(schema: dict[str, Any]) -> float:
    ratio_obj = _find_obj_by_alias(
        schema,
        ("c_ratio", "capacitance_ratio", "c2/c1", "cf/ci", "c_final/c_initial"),
        obj_type="ratio",
        role="given",
    )
    if ratio_obj is not None:
        return _dimensionless_value(ratio_obj)

    final_obj = _find_obj_by_alias(
        schema,
        ("c_final", "cf", "final_capacitance"),
        obj_type="capacitance",
        role="given",
    )
    initial_obj = _find_obj_by_alias(
        schema,
        ("c_initial", "ci", "initial_capacitance"),
        obj_type="capacitance",
        role="given",
    )

    if final_obj is not None and initial_obj is not None:
        return _to_si_quantity(final_obj, "F") / _to_si_quantity(initial_obj, "F")

    # fallback: n?u schema c? ??ng 2 capacitance given, hi?u object th? 2 l? final.
    caps = _objects_by_type(schema, "capacitance", role="given")
    if len(caps) >= 2:
        return _to_si_quantity(caps[1], "F") / _to_si_quantity(caps[0], "F")

    raise ValueError("Missing capacitance ratio or initial/final capacitance.")


def _voltage_ratio(schema: dict[str, Any]) -> float:
    ratio_obj = _find_obj_by_alias(
        schema,
        ("v_ratio", "u_ratio", "voltage_ratio", "v2/v1", "u2/u1"),
        obj_type="ratio",
        role="given",
    )
    if ratio_obj is not None:
        return _dimensionless_value(ratio_obj)

    volts = _objects_by_type(schema, "voltage", role="given")
    if len(volts) >= 2:
        return _to_si_quantity(volts[1], "V") / _to_si_quantity(volts[0], "V")

    raise ValueError("Missing voltage ratio or two voltage values.")


def _charge_ratio(schema: dict[str, Any]) -> float:
    ratio_obj = _find_obj_by_alias(
        schema,
        ("q_ratio", "charge_ratio", "q2/q1"),
        obj_type="ratio",
        role="given",
    )
    if ratio_obj is not None:
        return _dimensionless_value(ratio_obj)

    charges = _objects_by_type(schema, "charge", role="given")
    if len(charges) >= 2:
        return _to_si_quantity(charges[1], "C") / _to_si_quantity(charges[0], "C")

    raise ValueError("Missing charge ratio or two charge values.")


def _solve_capacitor_energy_scaling_constant_voltage(schema: dict[str, Any], formula: str) -> SolveResult:
    try:
        ratio = _capacitance_ratio(schema)
        query = _query_obj(schema, "ratio") or _query_obj(schema)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step(
        "Formula selected",
        "At constant voltage, W = 1/2*C*U^2, so energy scales proportional to capacitance.",
    )
    result.answer = _answer_ratio(ratio, query)
    return result


def _solve_capacitor_energy_voltage_scaling_constant_capacitance(schema: dict[str, Any], formula: str) -> SolveResult:
    try:
        v_ratio = _voltage_ratio(schema)
        w_ratio = v_ratio * v_ratio
        query = _query_obj(schema, "ratio") or _query_obj(schema)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step(
        "Formula selected",
        "At constant capacitance, W = 1/2*C*U^2, so W2/W1 = (U2/U1)^2.",
    )
    result.answer = _answer_ratio(w_ratio, query)
    return result


def _solve_capacitor_energy_charge_scaling_constant_capacitance(schema: dict[str, Any], formula: str) -> SolveResult:
    try:
        q_ratio = _charge_ratio(schema)
        w_ratio = q_ratio * q_ratio
        query = _query_obj(schema, "ratio") or _query_obj(schema)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step(
        "Formula selected",
        "At constant capacitance, W = Q^2/(2C), so W2/W1 = (Q2/Q1)^2.",
    )
    result.answer = _answer_ratio(w_ratio, query)
    return result


def _solve_series_capacitance_unknown(schema: dict[str, Any], formula: str) -> SolveResult:
    try:
        ceq_obj = _find_obj_by_alias(
            schema,
            ("ceq", "c_eq", "equivalent", "total"),
            obj_type="capacitance",
        )
        if ceq_obj is None:
            raise ValueError("Missing equivalent capacitance Ceq.")

        known_caps = [
            obj
            for obj in _objects_by_type(schema, "capacitance")
            if obj.get("role") in {"given", "constant"}
            and obj is not ceq_obj
            and "equivalent" not in _id_symbol_text(obj)
            and "ceq" not in _id_symbol_text(obj)
            and "c_eq" not in _id_symbol_text(obj)
        ]

        if not known_caps:
            raise ValueError("Missing known series capacitance.")

        c_known_obj = known_caps[0]
        query = _query_obj(schema, "capacitance")

        ceq = _to_si_quantity(ceq_obj, "F")
        c_known = _to_si_quantity(c_known_obj, "F")

        if ceq <= 0 or c_known <= 0:
            raise ValueError("Capacitances must be positive.")

        denominator = (1.0 / ceq) - (1.0 / c_known)
        if denominator <= 0:
            raise ValueError("Invalid series setup: need Ceq < known capacitor for positive unknown capacitance.")

        c_unknown = 1.0 / denominator
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step(
        "Formula selected",
        "For two capacitors in series, 1/Ceq = 1/C1 + 1/C2.",
    )
    result.answer = _answer(c_unknown, query, "capacitance", _raw_unit(c_known_obj, "F"))
    return result


def _solve_inductor_energy(schema: dict[str, Any], formula: str) -> SolveResult:
    w_query = _query_obj(schema, "energy")
    l_query = _query_obj(schema, "inductance")
    i_query = _query_obj(schema, "current")

    try:
        if w_query is not None:
            l = _required_si(schema, "inductance", "H")
            i = _required_si(schema, "current", "A")
            value = 0.5 * l * i * i
            query = w_query
            quantity_type = "energy"
            default_unit = "J"
        elif i_query is not None:
            w = _required_si(schema, "energy", "J")
            l = _required_si(schema, "inductance", "H")
            if l == 0:
                return _fail("Inductance must be non-zero when solving current.", formula)
            value = math.sqrt(2 * w / l)
            query = i_query
            quantity_type = "current"
            default_unit = "A"
        elif l_query is not None:
            w = _required_si(schema, "energy", "J")
            i = _required_si(schema, "current", "A")
            if i == 0:
                return _fail("Current must be non-zero when solving inductance.", formula)
            value = 2 * w / (i * i)
            query = l_query
            quantity_type = "inductance"
            default_unit = "H"
        else:
            return _fail("No supported query object for W = 1/2*L*I^2.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use inductor magnetic energy W = 1/2*L*I^2.")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_lc_resonance_frequency(schema: dict[str, Any], formula: str) -> SolveResult:
    f_query = _query_obj(schema, "frequency")
    c_query = _query_obj(schema, "capacitance")
    l_query = _query_obj(schema, "inductance")

    try:
        if f_query is not None:
            l = _required_si(schema, "inductance", "H")
            c = _required_si(schema, "capacitance", "F")
            if l <= 0 or c <= 0:
                return _fail("L and C must be positive.", formula)
            value = 1.0 / (2.0 * math.pi * math.sqrt(l * c))
            query = f_query
            quantity_type = "frequency"
            default_unit = "Hz"
        elif c_query is not None:
            f = _required_si(schema, "frequency", "Hz")
            l = _required_si(schema, "inductance", "H")
            if f <= 0 or l <= 0:
                return _fail("f and L must be positive.", formula)
            value = 1.0 / ((2.0 * math.pi * f) ** 2 * l)
            query = c_query
            quantity_type = "capacitance"
            default_unit = "uF"
        elif l_query is not None:
            f = _required_si(schema, "frequency", "Hz")
            c = _required_si(schema, "capacitance", "F")
            if f <= 0 or c <= 0:
                return _fail("f and C must be positive.", formula)
            value = 1.0 / ((2.0 * math.pi * f) ** 2 * c)
            query = l_query
            quantity_type = "inductance"
            default_unit = "H"
        else:
            return _fail("No supported query object for LC resonance frequency.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use LC resonance formula f = 1/(2*pi*sqrt(L*C)).")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_lc_resonance_angular_frequency(schema: dict[str, Any], formula: str) -> SolveResult:
    omega_query = _query_obj(schema, "angular_frequency")
    c_query = _query_obj(schema, "capacitance")
    l_query = _query_obj(schema, "inductance")

    try:
        if omega_query is not None:
            l = _required_si(schema, "inductance", "H")
            c = _required_si(schema, "capacitance", "F")
            if l <= 0 or c <= 0:
                return _fail("L and C must be positive.", formula)
            value = 1.0 / math.sqrt(l * c)
            query = omega_query
            quantity_type = "angular_frequency"
            default_unit = "rad/s"
        elif c_query is not None:
            omega = _required_si(schema, "angular_frequency", "rad/s")
            l = _required_si(schema, "inductance", "H")
            if omega <= 0 or l <= 0:
                return _fail("omega and L must be positive.", formula)
            value = 1.0 / (omega * omega * l)
            query = c_query
            quantity_type = "capacitance"
            default_unit = "uF"
        elif l_query is not None:
            omega = _required_si(schema, "angular_frequency", "rad/s")
            c = _required_si(schema, "capacitance", "F")
            if omega <= 0 or c <= 0:
                return _fail("omega and C must be positive.", formula)
            value = 1.0 / (omega * omega * c)
            query = l_query
            quantity_type = "inductance"
            default_unit = "H"
        else:
            return _fail("No supported query object for LC angular resonance.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use LC angular resonance formula omega = 1/sqrt(L*C).")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_ohm_law(schema: dict[str, Any], formula: str) -> SolveResult:
    u_query = _query_obj(schema, "voltage")
    i_query = _query_obj(schema, "current")
    r_query = _query_obj(schema, "resistance")

    try:
        if u_query is not None:
            i = _required_si(schema, "current", "A")
            r = _required_si(schema, "resistance", "?")
            value = i * r
            query = u_query
            quantity_type = "voltage"
            default_unit = "V"
        elif i_query is not None:
            u = _required_si(schema, "voltage", "V")
            r = _required_si(schema, "resistance", "?")
            if r == 0:
                return _fail("Resistance must be non-zero when solving current.", formula)
            value = u / r
            query = i_query
            quantity_type = "current"
            default_unit = "A"
        elif r_query is not None:
            u = _required_si(schema, "voltage", "V")
            i = _required_si(schema, "current", "A")
            if i == 0:
                return _fail("Current must be non-zero when solving resistance.", formula)
            value = u / i
            query = r_query
            quantity_type = "resistance"
            default_unit = "?"
        else:
            return _fail("No supported query object for U = I*R.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use Ohm's law U = I*R.")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_impedance_voltage_current(schema: dict[str, Any], formula: str) -> SolveResult:
    u_query = _query_obj(schema, "voltage")
    i_query = _query_obj(schema, "current")
    z_query = _query_obj(schema, "impedance")

    try:
        if u_query is not None:
            i = _required_si(schema, "current", "A")
            z = _required_si(schema, "impedance", "?")
            value = i * z
            query = u_query
            quantity_type = "voltage"
            default_unit = "V"
        elif i_query is not None:
            u = _required_si(schema, "voltage", "V")
            z = _required_si(schema, "impedance", "?")
            if z == 0:
                return _fail("Impedance must be non-zero when solving current.", formula)
            value = u / z
            query = i_query
            quantity_type = "current"
            default_unit = "A"
        elif z_query is not None:
            u = _required_si(schema, "voltage", "V")
            i = _required_si(schema, "current", "A")
            if i == 0:
                return _fail("Current must be non-zero when solving impedance.", formula)
            value = u / i
            query = z_query
            quantity_type = "impedance"
            default_unit = "?"
        else:
            return _fail("No supported query object for U = I*Z.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use impedance relation U = I*Z.")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_power_voltage_current(schema: dict[str, Any], formula: str) -> SolveResult:
    p_query = _query_obj(schema, "power")
    u_query = _query_obj(schema, "voltage")
    i_query = _query_obj(schema, "current")

    try:
        if p_query is not None:
            u = _required_si(schema, "voltage", "V")
            i = _required_si(schema, "current", "A")
            value = u * i
            query = p_query
            quantity_type = "power"
            default_unit = "W"
        elif u_query is not None:
            p = _required_si(schema, "power", "W")
            i = _required_si(schema, "current", "A")
            if i == 0:
                return _fail("Current must be non-zero when solving voltage.", formula)
            value = p / i
            query = u_query
            quantity_type = "voltage"
            default_unit = "V"
        elif i_query is not None:
            p = _required_si(schema, "power", "W")
            u = _required_si(schema, "voltage", "V")
            if u == 0:
                return _fail("Voltage must be non-zero when solving current.", formula)
            value = p / u
            query = i_query
            quantity_type = "current"
            default_unit = "A"
        else:
            return _fail("No supported query object for P = U*I.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use electric power formula P = U*I.")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_power_voltage_resistance(schema: dict[str, Any], formula: str) -> SolveResult:
    p_query = _query_obj(schema, "power")
    u_query = _query_obj(schema, "voltage")
    r_query = _query_obj(schema, "resistance")

    try:
        if p_query is not None:
            u = _required_si(schema, "voltage", "V")
            r = _required_si(schema, "resistance", "?")
            if r == 0:
                return _fail("Resistance must be non-zero when solving power.", formula)
            value = u * u / r
            query = p_query
            quantity_type = "power"
            default_unit = "W"
        elif u_query is not None:
            p = _required_si(schema, "power", "W")
            r = _required_si(schema, "resistance", "?")
            value = math.sqrt(p * r)
            query = u_query
            quantity_type = "voltage"
            default_unit = "V"
        elif r_query is not None:
            u = _required_si(schema, "voltage", "V")
            p = _required_si(schema, "power", "W")
            if p == 0:
                return _fail("Power must be non-zero when solving resistance.", formula)
            value = u * u / p
            query = r_query
            quantity_type = "resistance"
            default_unit = "?"
        else:
            return _fail("No supported query object for P = U^2/R.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use resistor/resonance power formula P = U^2/R.")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_power_current_resistance(schema: dict[str, Any], formula: str) -> SolveResult:
    p_query = _query_obj(schema, "power")
    i_query = _query_obj(schema, "current")
    r_query = _query_obj(schema, "resistance")

    try:
        if p_query is not None:
            i = _required_si(schema, "current", "A")
            r = _required_si(schema, "resistance", "?")
            value = i * i * r
            query = p_query
            quantity_type = "power"
            default_unit = "W"
        elif i_query is not None:
            p = _required_si(schema, "power", "W")
            r = _required_si(schema, "resistance", "?")
            if r == 0:
                return _fail("Resistance must be non-zero when solving current.", formula)
            value = math.sqrt(p / r)
            query = i_query
            quantity_type = "current"
            default_unit = "A"
        elif r_query is not None:
            p = _required_si(schema, "power", "W")
            i = _required_si(schema, "current", "A")
            if i == 0:
                return _fail("Current must be non-zero when solving resistance.", formula)
            value = p / (i * i)
            query = r_query
            quantity_type = "resistance"
            default_unit = "?"
        else:
            return _fail("No supported query object for P = I^2*R.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use resistor power formula P = I^2*R.")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_ac_impedance(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "impedance")
    if query is None:
        return _fail("Only impedance query is supported for Z = sqrt(R^2 + (XL-XC)^2).", formula)

    try:
        r = _required_si(schema, "resistance", "?")
        xl = _required_si(schema, "inductive_reactance", "?")
        xc = _required_si(schema, "capacitive_reactance", "?")
        value = math.sqrt(r * r + (xl - xc) * (xl - xc))
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use series RLC impedance Z = sqrt(R^2 + (XL-XC)^2).")
    result.answer = _answer(value, query, "impedance", "?")
    return result


def _solve_power_factor(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, ("power_factor", "ratio")) or _query_obj(schema)
    try:
        r = _required_si(schema, "resistance", "?")
        z = _required_si(schema, "impedance", "?")
        if z == 0:
            return _fail("Impedance must be non-zero when solving power factor.", formula)
        value = r / z
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use power factor cos(phi) = R/Z.")
    result.answer = _answer(value, query, "power_factor", "-")
    return result


def _solve_quality_factor(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, ("quality_factor", "ratio")) or _query_obj(schema)
    try:
        l = _required_si(schema, "inductance", "H")
        c = _required_si(schema, "capacitance", "F")
        r = _required_si(schema, "resistance", "?")
        if c <= 0 or r == 0:
            return _fail("C must be positive and R must be non-zero.", formula)
        value = math.sqrt(l / c) / r
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use series RLC quality factor Q = sqrt(L/C)/R.")
    result.answer = _answer(value, query, "quality_factor", "-")
    return result


def _solve_frequency_scaling_for_resonance(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, ("frequency_factor", "ratio")) or _query_obj(schema)

    try:
        xl = _required_si(schema, "inductive_reactance", "?")
        xc = _required_si(schema, "capacitive_reactance", "?")
        if xl <= 0 or xc <= 0:
            return _fail("XL and XC must be positive.", formula)
        value = math.sqrt(xc / xl)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step(
        "Formula selected",
        "Since XL is proportional to omega and XC is proportional to 1/omega, resonance factor k = sqrt(XC/XL).",
    )
    result.answer = _answer_ratio(value, query)
    return result


MU0 = 4 * math.pi * 1e-7


def _solve_solenoid_turn_density(schema: dict[str, Any], formula: str) -> SolveResult:
    n_query = _query_obj(schema, "turn_density")
    n_turns_query = _query_obj(schema, "turn_count")
    length_query = _query_obj(schema, ("length", "distance"))

    try:
        if n_query is not None:
            n_turns = _required_si(schema, "turn_count", "turns")
            length = _required_si(schema, ("length", "distance"), "m")
            if length == 0:
                return _fail("Length must be non-zero when solving turn density.", formula)
            value = n_turns / length
            query = n_query
            quantity_type = "turn_density"
            default_unit = "turns/m"
        elif n_turns_query is not None:
            n_density = _required_si(schema, "turn_density", "turns/m")
            length = _required_si(schema, ("length", "distance"), "m")
            value = n_density * length
            query = n_turns_query
            quantity_type = "turn_count"
            default_unit = "turns"
        elif length_query is not None:
            n_turns = _required_si(schema, "turn_count", "turns")
            n_density = _required_si(schema, "turn_density", "turns/m")
            if n_density == 0:
                return _fail("Turn density must be non-zero when solving length.", formula)
            value = n_turns / n_density
            query = length_query
            quantity_type = "length"
            default_unit = "m"
        else:
            return _fail("No supported query object for n = N/l.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use solenoid turn density n = N/l.")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_solenoid_magnetic_field(schema: dict[str, Any], formula: str) -> SolveResult:
    b_query = _query_obj(schema, "magnetic_field")
    i_query = _query_obj(schema, "current")

    try:
        n_turns_obj = _given_obj(schema, "turn_count")
        length_obj = _given_obj(schema, ("length", "distance"))
        n_density_obj = _given_obj(schema, "turn_density")

        if n_density_obj is not None:
            n_density = _to_si_quantity(n_density_obj, "turns/m")
        elif n_turns_obj is not None and length_obj is not None:
            n_turns = _to_si_quantity(n_turns_obj, "turns")
            length = _to_si_quantity(length_obj, "m")
            if length == 0:
                return _fail("Length must be non-zero.", formula)
            n_density = n_turns / length
        else:
            return _fail("Missing turn count + length, or turn density.", formula)

        if b_query is not None:
            current = _required_si(schema, "current", "A")
            value = MU0 * n_density * current
            query = b_query
            quantity_type = "magnetic_field"
            default_unit = "T"
        elif i_query is not None:
            b = _required_si(schema, "magnetic_field", "T")
            if n_density == 0:
                return _fail("Turn density must be non-zero when solving current.", formula)
            value = b / (MU0 * n_density)
            query = i_query
            quantity_type = "current"
            default_unit = "A"
        else:
            return _fail("No supported query object for B = mu0*n*I.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use long-solenoid magnetic field B = mu0*n*I = mu0*N*I/l.")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_solenoid_inductance(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "inductance")
    if query is None:
        return _fail("Only inductance query is supported for L = mu0*N^2*A/l.", formula)

    try:
        n_turns = _required_si(schema, "turn_count", "turns")
        area = _area_si(schema)
        length = _required_si(schema, ("length", "distance"), "m")
        if length == 0:
            return _fail("Length must be non-zero.", formula)
        value = MU0 * n_turns * n_turns * area / length
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use ideal solenoid inductance L = mu0*N^2*A/l.")
    result.answer = _answer(value, query, "inductance", "mH")
    return result


def _solve_magnetic_flux_total(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "magnetic_flux")
    if query is None:
        return _fail("Only magnetic flux query is supported for Phi = N*B*A.", formula)

    try:
        n_obj = _given_obj(schema, "turn_count")
        n_turns = _to_si_quantity(n_obj, "turns") if n_obj is not None else 1.0
        b = _required_si(schema, "magnetic_field", "T")
        area = _area_si(schema)
        value = n_turns * b * area
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use total flux linkage Phi = N*B*A.")
    result.answer = _answer(value, query, "magnetic_flux", "Wb")
    return result


def _solve_magnetic_energy_density(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "energy_density")
    if query is None:
        return _fail("Only energy density query is supported for u = B^2/(2*mu0).", formula)

    try:
        b = _required_si(schema, "magnetic_field", "T")
        value = b * b / (2.0 * MU0)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use magnetic field energy density u = B^2/(2*mu0).")
    result.answer = _answer(value, query, "energy_density", "J/m3")
    return result

def solve_schema(schema: dict[str, Any]) -> SolveResult:
    formula = _formula_id(schema)

    handlers = {
        "capacitor_charge_voltage": _solve_capacitor_charge_voltage,
        "capacitor_energy_voltage": _solve_capacitor_energy_voltage,
        "parallel_plate_capacitance": _solve_parallel_plate_capacitance,
        "parallel_plate_charge_from_voltage": _solve_parallel_plate_charge_from_voltage,
        "parallel_plate_charge_from_field": _solve_parallel_plate_charge_from_field,
        "parallel_plate_field": _solve_parallel_plate_field,
        "capacitor_energy_density": _solve_capacitor_energy_density,
        "capacitor_energy_scaling_constant_voltage": _solve_capacitor_energy_scaling_constant_voltage,
        "energy_scaling_constant_voltage": _solve_capacitor_energy_scaling_constant_voltage,
        "capacitor_energy_constant_voltage_scaling": _solve_capacitor_energy_scaling_constant_voltage,
        "capacitor_energy_voltage_scaling_constant_capacitance": _solve_capacitor_energy_voltage_scaling_constant_capacitance,
        "capacitor_energy_charge_scaling_constant_capacitance": _solve_capacitor_energy_charge_scaling_constant_capacitance,
        "series_capacitance_unknown": _solve_series_capacitance_unknown,
        "capacitor_series_unknown": _solve_series_capacitance_unknown,
        "inductor_energy": _solve_inductor_energy,
        "lc_magnetic_energy_inductor": _solve_inductor_energy,
        "lc_resonance_frequency": _solve_lc_resonance_frequency,
        "lc_resonance_angular_frequency": _solve_lc_resonance_angular_frequency,
        "resonant_impedance_equals_resistance": _solve_ohm_law,
        "ohm_law": _solve_ohm_law,
        "impedance_voltage_current": _solve_impedance_voltage_current,
        "power_voltage_current": _solve_power_voltage_current,
        "power_voltage_resistance": _solve_power_voltage_resistance,
        "power_current_resistance": _solve_power_current_resistance,
        "ac_impedance": _solve_ac_impedance,
        "power_factor": _solve_power_factor,
        "quality_factor": _solve_quality_factor,
        "frequency_scaling_for_resonance": _solve_frequency_scaling_for_resonance,
        "solenoid_turn_density": _solve_solenoid_turn_density,
        "solenoid_magnetic_field": _solve_solenoid_magnetic_field,
        "solenoid_magnetic_field_from_n": _solve_solenoid_magnetic_field,
        "solenoid_inductance": _solve_solenoid_inductance,
        "magnetic_flux_total": _solve_magnetic_flux_total,
        "magnetic_energy_density": _solve_magnetic_energy_density,
    }

    handler = handlers.get(formula)
    if handler is None:
        result = _new_result("unsupported")
        result.error = f"Unsupported equation formula: {formula or '<missing>'}"
        result.add_step(
            "Equation unsupported",
            f"No equations handler is registered for formula '{formula or '<missing>'}'.",
        )
        return result

    return handler(schema, formula)


def solve(question: str) -> SolveResult:
    result = _new_result("unsupported")
    result.add_step(
        "Domain selected",
        "The router selected the scalar equation solver.",
    )
    result.error = "Equation text adapter has not been migrated yet. Use solve_schema(schema)."
    return result
