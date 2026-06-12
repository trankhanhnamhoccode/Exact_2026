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
