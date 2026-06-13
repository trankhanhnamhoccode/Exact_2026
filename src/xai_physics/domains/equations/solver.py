from __future__ import annotations

import ast
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


_ALLOWED_EXPR_NAMES = {
    "pi": math.pi,
    "e": math.e,
}


def _safe_numeric_expr(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Could not parse numeric value: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        pass

    # LLM sometimes emits area as "3.14159 * (0.6)^2".
    expr = text.replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Could not parse numeric value: {value!r}") from exc

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in _ALLOWED_EXPR_NAMES:
            return float(_ALLOWED_EXPR_NAMES[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = eval_node(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a = eval_node(node.left)
            b = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow):
                return a ** b
        raise ValueError(f"Could not parse numeric value: {value!r}")

    return eval_node(tree)






def _unit_factor(unit: str | None) -> float:
    if unit is None:
        return 1.0

    raw = str(unit).strip().replace(" ", "")

    # IMPORTANT:
    # Check raw resistance units before normalize_unit().
    # Some text normalization paths can turn Greek Omega "?" into a degree-like token,
    # which makes 40 ? behave like 40 degrees = 0.698 rad.
    raw_aliases = {
        "?": 1.0,
        "?": 1.0,
        "ohm": 1.0,
        "ohms": 1.0,
    }
    if raw in raw_aliases:
        return raw_aliases[raw]

    normalized = normalize_unit(raw)
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

        # charge / line charge density
        "C": 1.0,
        "mC": 1e-3,
        "uC": 1e-6,
        "?C": 1e-6,
        "?C": 1e-6,
        "?C": 1e-6,
        "nC": 1e-9,
        "pC": 1e-12,
        "C/m": 1.0,
        "mC/m": 1e-3,
        "uC/m": 1e-6,
        "?C/m": 1e-6,
        "?C/m": 1e-6,
        "?C/m": 1e-6,
        "nC/m": 1e-9,

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

        # mass / force / acceleration
        "kg": 1.0,
        "g": 1e-3,
        "N": 1.0,
        "mN": 1e-3,
        "uN": 1e-6,
        "?N": 1e-6,
        "?N": 1e-6,
        "?N": 1e-6,
        "m/s2": 1.0,
        "m/s^2": 1.0,
        "m/s?": 1.0,

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
        "km": 1e3,
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
        "kV/m": 1e3,
        "N/C": 1.0,
        "ml": 1e-6,
        "mL": 1e-6,
        "L": 1e-3,
        "°C": 1.0,
        "Celsius": 1.0,

        # angle
        "rad": 1.0,
        "deg": math.pi / 180.0,
        "degree": math.pi / 180.0,
        "degrees": math.pi / 180.0,
        "?": math.pi / 180.0,
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

    numeric = _safe_numeric_expr(value)
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


def _has_value(obj: dict[str, Any] | None) -> bool:
    if not isinstance(obj, dict):
        return False
    return _raw_value(obj) is not None


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


def _iter_unique_objects(schema: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for obj in _all_relevant_objects(schema) + _objects(schema):
        if id(obj) in seen:
            continue
        out.append(obj)
        seen.add(id(obj))
    return out


def _type_matches(obj: dict[str, Any], obj_type: str | tuple[str, ...] | None) -> bool:
    if obj_type is None:
        return True
    types = (obj_type,) if isinstance(obj_type, str) else obj_type
    return obj.get("type") in types


def _query_obj(schema: dict[str, Any], obj_type: str | tuple[str, ...] | None = None) -> dict[str, Any] | None:
    candidates = [obj for obj in _iter_unique_objects(schema) if _type_matches(obj, obj_type)]

    # First prefer explicit query objects.
    for obj in candidates:
        if obj.get("role") == "query":
            return obj

    # LLM often writes role="unknown" for the thing being asked.
    for obj in candidates:
        if obj.get("role") == "unknown" and _raw_value(obj) is None:
            return obj

    # If an id/name says query and value is null, treat as query even if role is bad.
    for obj in candidates:
        blob = _id_symbol_text(obj) if "_id_symbol_text" in globals() else " ".join(str(obj.get(k, "")) for k in ("id", "symbol", "type")).lower()
        if "query" in blob and _raw_value(obj) is None:
            return obj

    return None


def _given_obj(schema: dict[str, Any], obj_type: str | tuple[str, ...]) -> dict[str, Any] | None:
    # Critical: do not treat value=None as a given. This was causing errors like
    # "Missing numeric value" when the LLM marked d1/f_query as role=given but left value null.
    for role in ("given", "constant"):
        for obj in _iter_unique_objects(schema):
            if _type_matches(obj, obj_type) and obj.get("role") == role and _has_value(obj):
                return obj
    return None


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


def _solve_capacitor_energy_charge_voltage(schema: dict[str, Any], formula: str) -> SolveResult:
    w_query = _query_obj(schema, "energy")
    q_query = _query_obj(schema, "charge")
    u_query = _query_obj(schema, "voltage")

    try:
        if w_query is not None:
            q = _required_si(schema, "charge", "C")
            u = _required_si(schema, "voltage", "V")
            value = 0.5 * q * u
            query = w_query
            quantity_type = "energy"
            default_unit = "J"
        elif q_query is not None:
            w = _required_si(schema, "energy", "J")
            u = _required_si(schema, "voltage", "V")
            if u == 0:
                return _fail("Voltage must be non-zero when solving charge.", formula)
            value = 2 * w / u
            query = q_query
            quantity_type = "charge"
            default_unit = "C"
        elif u_query is not None:
            w = _required_si(schema, "energy", "J")
            q = _required_si(schema, "charge", "C")
            if q == 0:
                return _fail("Charge must be non-zero when solving voltage.", formula)
            value = 2 * w / q
            query = u_query
            quantity_type = "voltage"
            default_unit = "V"
        else:
            return _fail("No supported query object for W = 0.5*Q*U.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use capacitor energy formula W = 1/2*Q*U.")
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


def _solve_capacitor_voltage_series(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "voltage")
    if query is None:
        return _fail("No voltage query object for series capacitor voltage division.", formula)

    try:
        rel = _first_formula_relation(schema)
        id_map = {obj.get("id"): obj for obj in _objects(schema)}
        rel_ids = [obj_id for obj_id in rel.get("objects", []) if obj_id in id_map]
        rel_objs = [id_map[obj_id] for obj_id in rel_ids]

        caps = [obj for obj in rel_objs if obj.get("type") == "capacitance" and obj.get("role") != "query"]
        if len(caps) < 2:
            caps = [obj for obj in _objects(schema) if obj.get("type") == "capacitance" and obj.get("role") != "query"]
        if len(caps) < 2:
            return _fail("Need two capacitances for series capacitor voltage division.", formula)

        voltages = [obj for obj in rel_objs if obj.get("type") == "voltage" and obj.get("role") != "query"]
        if not voltages:
            voltages = [obj for obj in _objects(schema) if obj.get("type") == "voltage" and obj.get("role") != "query"]
        if not voltages:
            return _fail("Need total voltage for series capacitor voltage division.", formula)

        c1 = _to_si_quantity(caps[0], "F")
        c2 = _to_si_quantity(caps[1], "F")
        u_total = _to_si_quantity(voltages[0], "V")
        if c1 <= 0 or c2 <= 0:
            return _fail("Capacitances must be positive.", formula)

        qid = str(query.get("id", "")).lower()
        # For two capacitors in series: U1 = U*C2/(C1+C2), U2 = U*C1/(C1+C2).
        if "2" in qid and "1" not in qid.replace("u2", ""):
            value = u_total * c1 / (c1 + c2)
        elif "1" in qid:
            value = u_total * c2 / (c1 + c2)
        else:
            # If the query id is generic, use the last capacitance mentioned in the relation.
            value = u_total * c1 / (c1 + c2)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use series capacitor voltage division: U_i = Q/C_i with common Q.")
    result.answer = _answer(value, query, "voltage", "V")
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


def _solve_parallel_plate_capacitance_distance_scaling(schema: dict[str, Any], formula: str) -> SolveResult:
    try:
        c_initial_obj = None
        for obj in _iter_unique_objects(schema):
            if obj.get("type") == "capacitance" and obj.get("role") in {"given", "constant"} and _has_value(obj):
                c_initial_obj = obj
                break
        if c_initial_obj is None:
            return _fail("Missing initial capacitance.", formula)
        c_initial = _to_si_quantity(c_initial_obj, "F")

        ratio_obj = None
        for obj in _iter_unique_objects(schema):
            blob = _id_symbol_text(obj)
            if obj.get("type") == "ratio" and _has_value(obj) and any(k in blob for k in ["d2/d1", "df/di", "distance", "separation", "d_ratio"]):
                ratio_obj = obj
                break
        if ratio_obj is None:
            ratio_obj = _given_obj(schema, "ratio")
        if ratio_obj is None:
            return _fail("Missing distance ratio d2/d1.", formula)
        d_ratio = _to_si_quantity(ratio_obj, "times")
        if d_ratio == 0:
            return _fail("Distance ratio must be non-zero.", formula)

        query = _query_obj(schema, "capacitance")
        if query is None:
            return _fail("Missing capacitance query.", formula)
        value = c_initial / d_ratio
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "For fixed area and medium, parallel-plate capacitance is inversely proportional to separation: C2 = C1/(d2/d1).")
    result.answer = _answer(value, query, "capacitance", _raw_unit(c_initial_obj, "F"))
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


K_COULOMB = 8.9875517923e9
G_DEFAULT = 10.0


def _required_scalar(schema: dict[str, Any], obj_type: str | tuple[str, ...], unit: str = "") -> float:
    obj = _given_obj(schema, obj_type)
    if obj is None:
        raise ValueError(f"Missing given object of type {obj_type}")
    return _to_si_quantity(obj, unit)


def _solve_point_charge_electric_field(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "electric_field")
    if query is None:
        return _fail("Only electric field query is supported for E = k|q|/(eps_r*r^2).", formula)

    try:
        q = _required_si(schema, "charge", "C")
        r = _required_si(schema, ("distance", "length", "radius"), "m")
        eps_r = _relative_permittivity(schema)
        if r == 0 or eps_r == 0:
            return _fail("Distance and relative permittivity must be non-zero.", formula)
        value = K_COULOMB * abs(q) / (eps_r * r * r)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use point-charge electric field E = k|q|/(eps_r*r^2).")
    result.answer = _answer(value, query, "electric_field", "V/m")
    return result


def _solve_electric_force_field(schema: dict[str, Any], formula: str) -> SolveResult:
    f_query = _query_obj(schema, "force")
    q_query = _query_obj(schema, "charge")
    e_query = _query_obj(schema, "electric_field")

    try:
        if f_query is not None:
            q = _required_si(schema, "charge", "C")
            e = _required_si(schema, "electric_field", "V/m")
            value = abs(q) * e
            query = f_query
            quantity_type = "force"
            default_unit = "N"
        elif q_query is not None:
            f = _required_si(schema, "force", "N")
            e = _required_si(schema, "electric_field", "V/m")
            if e == 0:
                return _fail("Electric field must be non-zero when solving charge.", formula)
            value = abs(f) / e
            query = q_query
            quantity_type = "charge"
            default_unit = "C"
        elif e_query is not None:
            f = _required_si(schema, "force", "N")
            q = _required_si(schema, "charge", "C")
            if q == 0:
                return _fail("Charge must be non-zero when solving electric field.", formula)
            value = abs(f) / abs(q)
            query = e_query
            quantity_type = "electric_field"
            default_unit = "V/m"
        else:
            return _fail("No supported query object for F = |q|E.", formula)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use electric force relation F = |q|E.")
    result.answer = _answer(value, query, quantity_type, default_unit)
    return result


def _solve_equilibrium_electric_field(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "electric_field")
    if query is None:
        return _fail("Only electric field query is supported for |q|E = mg.", formula)

    try:
        mass = _required_si(schema, "mass", "kg")
        charge = _required_si(schema, "charge", "C")

        g_obj = _given_obj(schema, ("acceleration", "gravitational_acceleration"))
        g = _to_si_quantity(g_obj, "m/s2") if g_obj is not None else G_DEFAULT

        if charge == 0:
            return _fail("Charge must be non-zero.", formula)

        value = mass * g / abs(charge)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "For vertical electric equilibrium, balance |q|E = mg.")
    result.answer = _answer(value, query, "electric_field", "V/m")
    return result


def _solve_infinite_wire_electric_field(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "electric_field")
    if query is None:
        return _fail("Only electric field query is supported for E = 2k|lambda|/(eps_r*r).", formula)

    try:
        line_charge = _required_si(schema, "line_charge_density", "C/m")
        r = _required_si(schema, ("distance", "length", "radius"), "m")
        eps_r = _relative_permittivity(schema)
        if r == 0 or eps_r == 0:
            return _fail("Distance and relative permittivity must be non-zero.", formula)

        value = 2.0 * K_COULOMB * abs(line_charge) / (eps_r * r)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use infinite-wire electric field E = 2k|lambda|/(eps_r*r).")
    result.answer = _answer(value, query, "electric_field", "V/m")
    return result


def _solve_percentage_relative_error(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, ("percent_error", "relative_error", "ratio")) or _query_obj(schema)
    if query is None:
        return _fail("No query object for percentage relative error.", formula)

    try:
        abs_error = _required_scalar(schema, "absolute_error")
        measured = _required_scalar(schema, "measured_value")
        if measured == 0:
            return _fail("Measured value must be non-zero.", formula)
        percent = abs(abs_error) / abs(measured) * 100.0
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use percentage relative error = absolute error / measured value * 100%.")
    result.answer = f"{_fmt(percent)} %"
    return result


def _solve_absolute_error_from_actual(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "absolute_error")
    if query is None:
        return _fail("Only absolute error query is supported for |actual - measured|.", formula)

    try:
        actual_obj = _given_obj(schema, "actual_value")
        measured_obj = _given_obj(schema, "measured_value")
        if actual_obj is None or measured_obj is None:
            raise ValueError("Missing actual_value and/or measured_value.")

        unit = _raw_unit(query, _raw_unit(actual_obj, _raw_unit(measured_obj, "")))
        actual = _to_si_quantity(actual_obj, unit)
        measured = _to_si_quantity(measured_obj, unit)
        value = abs(actual - measured)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use absolute error = |actual value - measured value|.")
    result.answer = _answer(value, query, "absolute_error", _raw_unit(query, ""))
    return result


def _solve_capacitor_plate_force_by_charge_area(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "force")
    if query is None:
        return _fail("Only force query is supported for F = Q^2/(2*eps0*eps_r*A).", formula)

    try:
        q = _required_si(schema, "charge", "C")
        area = _area_si(schema)
        eps_r = _relative_permittivity(schema)
        if area <= 0 or eps_r == 0:
            return _fail("Area must be positive and relative permittivity must be non-zero.", formula)

        value = q * q / (2.0 * EPS0 * eps_r * area)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step(
        "Formula selected",
        "Use capacitor plate attraction force F = Q^2/(2*eps0*eps_r*A).",
    )
    result.answer = _answer(value, query, "force", "N")
    return result


def _solve_self_inductance_from_emf(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "inductance")
    if query is None:
        return _fail("Only inductance query is supported for L = emf*dt/dI.", formula)

    try:
        emf = _required_si(schema, "voltage", "V")
        dt = _required_si(schema, "time", "s")
        d_i = _required_si(schema, "current_change", "A")
        if d_i == 0:
            return _fail("Current change must be non-zero.", formula)

        value = abs(emf) * dt / abs(d_i)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step(
        "Formula selected",
        "Use self-induction relation |emf| = L*|dI|/dt, so L = |emf|*dt/|dI|.",
    )
    result.answer = _answer(value, query, "inductance", "H")
    return result


def _solve_equilibrium_mass_with_angle(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "mass")
    if query is None:
        return _fail("Only mass query is supported for m = |q|E/(g*tan(theta)).", formula)

    try:
        q = _required_si(schema, "charge", "C")
        e = _required_si(schema, "electric_field", "V/m")
        theta = _required_si(schema, "angle", "rad")

        g_obj = _given_obj(schema, ("acceleration", "gravitational_acceleration"))
        g = _to_si_quantity(g_obj, "m/s2") if g_obj is not None else G_DEFAULT

        tan_theta = math.tan(theta)
        if g == 0 or tan_theta == 0:
            return _fail("g and tan(theta) must be non-zero.", formula)

        value = abs(q) * e / (g * tan_theta)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step(
        "Formula selected",
        "For equilibrium with a string angle theta from vertical, tan(theta)=|q|E/(mg).",
    )
    result.answer = _answer(value, query, "mass", "kg")
    return result


def _solve_lc_current_amplitude_from_charge_amplitude(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, ("current_amplitude", "current"))
    if query is None:
        return _fail("Only current amplitude query is supported for I0 = omega*Q0.", formula)

    try:
        omega = _required_si(schema, "angular_frequency", "rad/s")
        q0_obj = _given_obj(schema, "charge_amplitude")
        if q0_obj is None:
            q0_obj = _given_obj(schema, "charge")
        if q0_obj is None:
            raise ValueError("Missing charge amplitude Q0.")

        q0 = _to_si_quantity(q0_obj, "C")
        value = omega * q0
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "In an LC circuit, current amplitude is I0 = omega*Q0.")
    result.answer = _answer(value, query, "current_amplitude", "A")
    return result


def _phase_value(schema: dict[str, Any]) -> float:
    obj = _given_obj(schema, ("phase", "phase_angle"))
    if obj is None:
        return 0.0
    return _to_si_quantity(obj, "rad")


def _omega_t_phi(schema: dict[str, Any]) -> float:
    omega = _required_si(schema, "angular_frequency", "rad/s")
    t = _required_si(schema, "time", "s")
    phi = _phase_value(schema)
    return omega * t + phi


def _solve_harmonic_current_cos_time(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, ("current", "current_amplitude"))
    if query is None:
        return _fail("Only current query is supported for i(t)=I0*cos(omega*t+phi).", formula)

    try:
        amp_obj = _given_obj(schema, "current_amplitude")
        if amp_obj is None:
            raise ValueError("Missing current amplitude I0.")
        amp = _to_si_quantity(amp_obj, "A")
        value = amp * math.cos(_omega_t_phi(schema))
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use sinusoidal current i(t)=I0*cos(omega*t+phi).")
    result.answer = _answer(value, query, "current", "A")
    return result


def _solve_harmonic_voltage_cos_time(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, ("voltage", "voltage_amplitude"))
    if query is None:
        return _fail("Only voltage query is supported for u(t)=U0*cos(omega*t+phi).", formula)

    try:
        amp_obj = _given_obj(schema, "voltage_amplitude")
        if amp_obj is None:
            raise ValueError("Missing voltage amplitude U0.")
        amp = _to_si_quantity(amp_obj, "V")
        value = amp * math.cos(_omega_t_phi(schema))
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use sinusoidal voltage u(t)=U0*cos(omega*t+phi).")
    result.answer = _answer(value, query, "voltage", "V")
    return result


def _solve_harmonic_charge_cos_time(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, ("charge", "charge_amplitude"))
    if query is None:
        return _fail("Only charge query is supported for q(t)=Q0*cos(omega*t+phi).", formula)

    try:
        amp_obj = _given_obj(schema, "charge_amplitude")
        if amp_obj is None:
            raise ValueError("Missing charge amplitude Q0.")
        amp = _to_si_quantity(amp_obj, "C")
        value = amp * math.cos(_omega_t_phi(schema))
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "Use sinusoidal charge q(t)=Q0*cos(omega*t+phi).")
    result.answer = _answer(value, query, "charge", "C")
    return result


def _solve_lc_electric_energy_time(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "energy")
    if query is None:
        return _fail("Only energy query is supported for We=Wtotal*cos^2(omega*t+phi).", formula)

    try:
        w_total = _required_si(schema, "energy", "J")
        c = math.cos(_omega_t_phi(schema))
        value = w_total * c * c
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "In an ideal LC circuit, electric energy We=Wtotal*cos^2(omega*t+phi).")
    result.answer = _answer(value, query, "energy", "J")
    return result


def _solve_lc_magnetic_energy_time(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "energy")
    if query is None:
        return _fail("Only energy query is supported for Wm=Wtotal*sin^2(omega*t+phi).", formula)

    try:
        w_total = _required_si(schema, "energy", "J")
        s = math.sin(_omega_t_phi(schema))
        value = w_total * s * s
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "In an ideal LC circuit, magnetic energy Wm=Wtotal*sin^2(omega*t+phi).")
    result.answer = _answer(value, query, "energy", "J")
    return result


def _solve_parallel_resistance(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, "resistance") or _query_obj(schema, "equivalent_resistance")
    if query is None:
        return _fail("Only resistance query is supported for parallel resistance.", formula)
    try:
        resistors = [obj for obj in _objects_by_type(schema, ("resistance", "equivalent_resistance")) if obj is not query and _has_value(obj)]
        if len(resistors) < 2:
            return _fail("Need at least two known resistances for parallel resistance.", formula)
        inv = 0.0
        for obj in resistors:
            r = _to_si_quantity(obj, "ohm")
            if r == 0:
                return _fail("Resistance must be non-zero.", formula)
            inv += 1.0 / r
        value = 1.0 / inv
    except Exception as exc:
        return _fail(str(exc), formula)
    result = _new_result()
    result.add_step("Formula selected", "Use parallel resistance 1/R = sum(1/R_i).")
    result.answer = _answer(value, query, "resistance", "ohm")
    return result


def _solve_power_factor_at_resonance(schema: dict[str, Any], formula: str) -> SolveResult:
    query = _query_obj(schema, ("power_factor", "ratio")) or _query_obj(schema)
    result = _new_result()
    result.add_step("Formula selected", "At resonance in a series RLC circuit, impedance is purely resistive, so cos(phi)=1.")
    result.answer = _answer(1.0, query, "power_factor", "")
    return result


def _solve_resonance_check(schema: dict[str, Any], formula: str) -> SolveResult:
    try:
        l = _required_si(schema, "inductance", "H")
        c = _required_si(schema, "capacitance", "F")
        f_obj = _given_obj(schema, "frequency")
        if f_obj is None:
            return _fail("Need supply frequency for resonance yes/no check.", formula)
        f = _to_si_quantity(f_obj, "Hz")
        f0 = 1.0 / (2.0 * math.pi * math.sqrt(l * c))
        # Dataset phrases round frequencies heavily, so allow absolute 0.6 Hz or 1.5% relative tolerance.
        ok = abs(f - f0) <= max(0.6, 0.015 * abs(f0))
    except Exception as exc:
        return _fail(str(exc), formula)
    result = _new_result()
    result.add_step("Formula selected", f"Check resonance by comparing f={f:g} Hz with f0={f0:g} Hz.")
    result.answer = "Yes" if ok else "No"
    return result


def _solve_instrument_absolute_error(schema: dict[str, Any], formula: str) -> SolveResult:
    try:
        least = _given_obj(schema, "least_count")
        if least is None:
            return _fail("Missing least count.", formula)
        query = _query_obj(schema, "absolute_error") or _query_obj(schema)
        unit = _raw_unit(query, _raw_unit(least, "")) if query is not None else _raw_unit(least, "")
        value = _to_si_quantity(least, unit)
    except Exception as exc:
        return _fail(str(exc), formula)

    result = _new_result()
    result.add_step("Formula selected", "For this dataset, absolute instrument error equals the least count.")
    result.answer = _answer(value, query, "absolute_error", unit)
    return result


def _solve_measurement_average(schema: dict[str, Any], formula: str) -> SolveResult:
    try:
        values = [
            obj for obj in _iter_unique_objects(schema)
            if obj.get("role") == "given"
            and obj.get("type") in {"measured_value", "length", "mass", "voltage", "current", "temperature", "volume"}
            and _has_value(obj)
        ]
        if not values:
            return _fail("Need measured values for average measurement.", formula)
        unit = _raw_unit(values[0], "")
        vals = [_to_si_quantity(obj, unit) for obj in values]
        mean = sum(vals) / len(vals)
        avg_abs_err = sum(abs(v - mean) for v in vals) / len(vals)
        queries = [obj for obj in _iter_unique_objects(schema) if obj.get("role") == "query"]
        if not queries:
            queries = [_query_obj(schema)]
        answers: list[str] = []
        for q in queries:
            if q is None:
                continue
            qtype = q.get("type")
            if qtype in {"average_value", "average_length", "mean_value", "mean"}:
                answers.append(_answer(mean, q, qtype, unit))
            elif qtype in {"average_absolute_error", "absolute_error"}:
                answers.append(_answer(avg_abs_err, q, qtype, unit))
            elif qtype in {"relative_error", "percent_error", "percentage_relative_error"}:
                if mean == 0:
                    return _fail("Mean must be non-zero for relative error.", formula)
                answers.append(f"{_fmt(100.0 * avg_abs_err / abs(mean))} %")
        if not answers:
            answers = [_answer(mean, queries[0] if queries else None, "average_value", unit)]
    except Exception as exc:
        return _fail(str(exc), formula)
    result = _new_result()
    result.add_step("Formula selected", "Compute mean and average absolute deviation from repeated measurements.")
    result.answer = "; ".join(answers)
    return result


def _canonical_formula(schema: dict[str, Any], formula: str) -> str:
    raw = (formula or "").strip()
    key = raw.lower()
    aliases = {
        "lc_resonant_frequency": "lc_resonance_frequency",
        "capacitor_energy_charge_voltage": "capacitor_energy_charge_voltage",
        "capacitor_energy_charge_voltage_formula": "capacitor_energy_charge_voltage",
        "energy_from_charge_voltage": "capacitor_energy_charge_voltage",
        "parallel_plate_capacitance_distance_scaling": "parallel_plate_capacitance_distance_scaling",
        "capacitance_distance_scaling": "parallel_plate_capacitance_distance_scaling",
        "plate_separation_capacitance_scaling": "parallel_plate_capacitance_distance_scaling",
        "least_count_absolute_error": "instrument_absolute_error",
        "instrument_absolute_error": "instrument_absolute_error",
        "least_count_error": "instrument_absolute_error",
        "capacitor_inductor_resonant_frequency": "lc_resonance_frequency",
        "resonant_frequency_inductor_capacitor": "lc_resonance_frequency",
        "resonance_frequency": "lc_resonance_frequency",
        "resonance_condition": "resonance_check",
        "power_factor_at_resonance": "power_factor_at_resonance",
        "resistance_voltage_current": "ohm_law",
        "voltage_current_resistance": "ohm_law",
        "power_from_voltage_current": "power_voltage_current",
        "percent_relative_uncertainty": "percentage_relative_error",
        "relative_uncertainty": "percentage_relative_error",
        "random_error": "measurement_average",
        "average_value": "measurement_average",
        "average_absolute_error": "measurement_average",
        "parallel_resistance": "parallel_resistance",
        "not_relevant": "capacitor_plate_force_by_charge_area",
        "turn_density": "solenoid_turn_density",
        "turn_density_from_turn_count_and_length": "solenoid_turn_density",
        "inductive_reactance": "ac_inductive_reactance",
        "capacitor_voltage_charge": "capacitor_charge_voltage",
        "total_impedance_from_components": "ac_impedance",
        "impedance_series": "ac_impedance",
    }
    key = aliases.get(key, key)

    # If the LLM picked frequency formula but the query is angular frequency, route to omega.
    if key == "lc_resonance_frequency" and _query_obj(schema, "angular_frequency") is not None:
        return "lc_resonance_angular_frequency"

    # If there is a given frequency and no numeric query, the question is often Yes/No resonance.
    if key in {"lc_resonance_frequency", "quality_factor"}:
        if _given_obj(schema, "frequency") is not None and _query_obj(schema, ("frequency", "angular_frequency", "capacitance", "inductance")) is None:
            return "resonance_check"
        # CHLT often extracts quality_factor query for yes/no resonance. Prefer check when f is present.
        if _given_obj(schema, "frequency") is not None and _query_obj(schema, "quality_factor") is not None:
            return "resonance_check"

    # Energy of capacitor with C and U sometimes gets mislabeled as energy_density.
    if key == "capacitor_energy_density":
        if _given_obj(schema, "capacitance") is not None and _given_obj(schema, "voltage") is not None and _query_obj(schema, "energy") is not None:
            return "capacitor_energy_voltage"

    # If final query is energy and Q + U are given, prefer direct W = 1/2 Q U even if the LLM emitted an intermediate C_query.
    if _query_obj(schema, "energy") is not None and _given_obj(schema, "charge") is not None and _given_obj(schema, "voltage") is not None:
        if key in {"capacitor_charge_voltage", "capacitor_energy_voltage", "capacitor_energy_charge_voltage"}:
            return "capacitor_energy_charge_voltage"

    # If any relation explicitly asks for average measurement, use the aggregate handler.
    names = {str(rel.get("name", "")).lower() for rel in _relations(schema)}
    if {"average_value", "average_absolute_error", "random_error"} & names:
        return "measurement_average"

    return key

def solve_schema(schema: dict[str, Any]) -> SolveResult:
    formula = _canonical_formula(schema, _formula_id(schema))

    handlers = {
        "capacitor_charge_voltage": _solve_capacitor_charge_voltage,
        "capacitor_energy_voltage": _solve_capacitor_energy_voltage,
        "capacitor_energy_charge_voltage": _solve_capacitor_energy_charge_voltage,
        "parallel_plate_capacitance": _solve_parallel_plate_capacitance,
        "parallel_plate_capacitance_distance_scaling": _solve_parallel_plate_capacitance_distance_scaling,
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
        "capacitor_voltage_series": _solve_capacitor_voltage_series,
        "series_capacitor_voltage_division": _solve_capacitor_voltage_series,
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
        "power_factor_at_resonance": _solve_power_factor_at_resonance,
        "quality_factor": _solve_quality_factor,
        "resonance_check": _solve_resonance_check,
        "frequency_scaling_for_resonance": _solve_frequency_scaling_for_resonance,
        "parallel_resistance": _solve_parallel_resistance,
        "instrument_absolute_error": _solve_instrument_absolute_error,
        "measurement_average": _solve_measurement_average,
        "solenoid_turn_density": _solve_solenoid_turn_density,
        "solenoid_magnetic_field": _solve_solenoid_magnetic_field,
        "solenoid_magnetic_field_from_n": _solve_solenoid_magnetic_field,
        "solenoid_inductance": _solve_solenoid_inductance,
        "magnetic_flux_total": _solve_magnetic_flux_total,
        "magnetic_energy_density": _solve_magnetic_energy_density,
        "point_charge_electric_field": _solve_point_charge_electric_field,
        "electric_force_field": _solve_electric_force_field,
        "equilibrium_electric_field": _solve_equilibrium_electric_field,
        "infinite_wire_electric_field": _solve_infinite_wire_electric_field,
        "percentage_relative_error": _solve_percentage_relative_error,
        "absolute_error_from_actual": _solve_absolute_error_from_actual,
        "capacitor_plate_force_by_charge_area": _solve_capacitor_plate_force_by_charge_area,
        "self_inductance_from_emf": _solve_self_inductance_from_emf,
        "equilibrium_mass_with_angle": _solve_equilibrium_mass_with_angle,
        "lc_current_amplitude_from_charge_amplitude": _solve_lc_current_amplitude_from_charge_amplitude,
        "harmonic_current_cos_time": _solve_harmonic_current_cos_time,
        "harmonic_voltage_cos_time": _solve_harmonic_voltage_cos_time,
        "harmonic_charge_cos_time": _solve_harmonic_charge_cos_time,
        "lc_electric_energy_time": _solve_lc_electric_energy_time,
        "lc_magnetic_energy_time": _solve_lc_magnetic_energy_time,
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
