from __future__ import annotations

from typing import Any

import sympy as sp

from xai_physics.core.answer import AnswerEnvelope
from xai_physics.core.result import SolveResult
from xai_physics.core.units import convert


class SymbolicGeometryError(ValueError):
    pass


_SYMBOL_ALIASES = {
    "ε": "epsilon",
    "ɛ": "epsilon",
    "ϵ": "epsilon",
    "λ": "lambda",
}


def _clean_expr_text(value: Any) -> str:
    text = str(value).strip()
    for raw, repl in _SYMBOL_ALIASES.items():
        text = text.replace(raw, repl)
    text = text.replace("^", "**")
    text = text.replace("×", "*").replace("·", "*")
    text = text.replace("−", "-")
    return text


def _symbols_for_text(text: str) -> dict[str, sp.Symbol]:
    names = set()
    for name in ["k", "q", "a", "h", "epsilon", "EA", "EB", "EM"]:
        if name in text:
            names.add(name)
    # Also collect simple identifiers that sympy can parse.
    import re
    for name in re.findall(r"[A-Za-z_]\w*", text):
        if name not in {"sqrt", "Abs"}:
            names.add(name)
    return {name: sp.Symbol(name, positive=True, real=True) for name in names}


def _parse_expr(value: Any, symbols: dict[str, sp.Symbol] | None = None) -> sp.Expr:
    text = _clean_expr_text(value)
    local_dict = dict(symbols or {})
    local_dict.setdefault("sqrt", sp.sqrt)
    local_dict.setdefault("Abs", sp.Abs)
    try:
        return sp.sympify(text, locals=local_dict)
    except Exception as exc:  # pragma: no cover - defensive message
        raise SymbolicGeometryError(f"Cannot parse symbolic expression {value!r}: {exc}") from exc


def _collect_symbols(schema: dict[str, Any]) -> dict[str, sp.Symbol]:
    chunks: list[str] = []
    for point in schema.get("points", []) or []:
        chunks.extend([str(point.get("x", "")), str(point.get("y", ""))])
    for charge in schema.get("charges", []) or []:
        chunks.append(str(charge.get("charge")))
    for query in schema.get("queries", []) or []:
        for key in ("variable", "unknown", "left", "right"):
            if key in query:
                chunks.append(str(query[key]))
    medium = schema.get("medium", {}) or {}
    for value in medium.values():
        chunks.append(str(value))
    text = " ".join(chunks)
    symbols = _symbols_for_text(text)
    for name in ["k", "q", "a", "h", "epsilon", "EA", "EB", "EM"]:
        symbols.setdefault(name, sp.Symbol(name, positive=True, real=True))
    return symbols


def _point_map(schema: dict[str, Any], symbols: dict[str, sp.Symbol]) -> dict[str, tuple[sp.Expr, sp.Expr]]:
    points: dict[str, tuple[sp.Expr, sp.Expr]] = {}
    for point in schema.get("points", []) or []:
        pid = str(point.get("id") or "").strip()
        if not pid:
            continue
        if "x" not in point or "y" not in point:
            raise SymbolicGeometryError(f"Point {pid} needs symbolic x/y coordinates.")
        points[pid] = (_parse_expr(point["x"], symbols), _parse_expr(point["y"], symbols))
    if not points:
        raise SymbolicGeometryError("symbolic_geometry schema needs points.")
    return points


def _charge_expr(charge: dict[str, Any], symbols: dict[str, sp.Symbol]) -> sp.Expr:
    raw = charge.get("charge")
    if isinstance(raw, dict):
        raw = raw.get("value")
    if raw is None:
        raise SymbolicGeometryError(f"Charge {charge.get('id')} needs a value/expression.")
    return _parse_expr(raw, symbols)


def _medium_denominator(schema: dict[str, Any], symbols: dict[str, sp.Symbol]) -> sp.Expr:
    medium = schema.get("medium", {}) or {}
    rel = medium.get("relative_permittivity_symbol") or medium.get("epsilon_symbol")
    if rel:
        return _parse_expr(rel, symbols)
    rel_value = medium.get("relative_permittivity")
    if rel_value is not None:
        return _parse_expr(rel_value, symbols)
    return sp.Integer(1)


def _field_vector(schema: dict[str, Any], target_id: str, symbols: dict[str, sp.Symbol]) -> tuple[sp.Expr, sp.Expr]:
    points = _point_map(schema, symbols)
    if target_id not in points:
        raise SymbolicGeometryError(f"Unknown target point {target_id}.")
    tx, ty = points[target_id]
    k = symbols.setdefault("k", sp.Symbol("k", positive=True, real=True))
    denom_medium = _medium_denominator(schema, symbols)

    ex = sp.Integer(0)
    ey = sp.Integer(0)
    for charge in schema.get("charges", []) or []:
        at = str(charge.get("at") or "")
        if at not in points:
            raise SymbolicGeometryError(f"Charge {charge.get('id')} references unknown point {at}.")
        sx, sy = points[at]
        dx = sp.simplify(tx - sx)
        dy = sp.simplify(ty - sy)
        r2 = sp.simplify(dx * dx + dy * dy)
        if r2 == 0:
            raise SymbolicGeometryError(f"Cannot compute field from charge {charge.get('id')} at its own point {target_id}.")
        q = _charge_expr(charge, symbols)
        coeff = k * q / (denom_medium * r2 ** sp.Rational(3, 2))
        ex += coeff * dx
        ey += coeff * dy
    return sp.simplify(ex), sp.simplify(ey)




def _result_domain(schema: dict[str, Any]) -> str:
    domain = str(schema.get("domain") or "symbolic_geometry")
    # New schemas should keep the physics domain and use solver_backend/subtype
    # for implementation routing.  Legacy p21/cache schemas still use the
    # top-level symbolic_geometry domain and remain supported.
    return "electrostatics" if domain == "electrostatics" else "symbolic_geometry"

def _magnitude_from_components(ex: sp.Expr, ey: sp.Expr) -> sp.Expr:
    ex = sp.simplify(ex)
    ey = sp.simplify(ey)
    if ex == 0:
        return sp.simplify(abs(ey))
    if ey == 0:
        return sp.simplify(abs(ex))
    return sp.simplify(sp.sqrt(ex * ex + ey * ey))


def _render(expr: sp.Expr) -> str:
    expr = sp.simplify(expr)
    s = sp.sstr(expr)
    s = s.replace("**", "^")
    s = s.replace("epsilon", "epsilon")
    return s


def _make_result(expr: sp.Expr, *, unit: str | None, step: str, domain: str = "symbolic_geometry") -> SolveResult:
    display = _render(expr)
    if unit:
        display = f"{display} {unit}"
    result = SolveResult(status="solved", domain=domain, answer=display)
    result.answer_meta = AnswerEnvelope.symbolic(
        display=display,
        canonical=("sympy", sp.sstr(sp.simplify(expr))),
        unit=unit,
        variables={str(s): str(s) for s in sorted(expr.free_symbols, key=lambda x: str(x))},
        formula="symbolic_geometry_vector_superposition",
        source="symbolic_geometry_solver",
    )
    result.add_step("Symbolic geometry", step, expression=sp.sstr(sp.simplify(expr)))
    return result


def _solve_electric_field(schema: dict[str, Any], query: dict[str, Any], symbols: dict[str, sp.Symbol]) -> SolveResult:
    target = str(query.get("target") or "")
    unit = query.get("unit") or "V/m"
    ex, ey = _field_vector(schema, target, symbols)
    output = str(query.get("output") or "magnitude")
    if output == "x_component":
        expr = ex
    elif output == "y_component":
        expr = ey
    elif output in {"components", "vector"}:
        display = {"Ex": f"{_render(ex)} {unit}", "Ey": f"{_render(ey)} {unit}"}
        result = SolveResult(status="solved", domain=_result_domain(schema), answer=display)
        result.add_step("Symbolic geometry", "Computed symbolic electric-field vector components.", Ex=sp.sstr(ex), Ey=sp.sstr(ey))
        return result
    else:
        expr = _magnitude_from_components(ex, ey)
    return _make_result(expr, unit=unit, step="Computed symbolic electric-field vector by superposition.", domain=_result_domain(schema))


def _solve_maximize_field(schema: dict[str, Any], query: dict[str, Any], symbols: dict[str, sp.Symbol]) -> SolveResult:
    target = str(query.get("target") or "")
    variable_name = str(query.get("variable") or "h")
    variable = symbols.setdefault(variable_name, sp.Symbol(variable_name, positive=True, real=True))
    ex, ey = _field_vector(schema, target, symbols)
    mag = _magnitude_from_components(ex, ey)
    derivative = sp.diff(mag, variable)
    solutions = sp.solve(sp.Eq(derivative, 0), variable)
    positive_solutions = [sp.simplify(sol) for sol in solutions if sol != 0]
    if not positive_solutions:
        raise SymbolicGeometryError(f"No symbolic optimum found for {variable_name}.")
    expr = positive_solutions[0]
    return _make_result(
        expr,
        unit=query.get("unit") or "m",
        step=f"Maximized symbolic field magnitude with respect to {variable_name}.",
        domain=_result_domain(schema),
    )


def _solve_zero_field_unknown_charge(schema: dict[str, Any], query: dict[str, Any], symbols: dict[str, sp.Symbol]) -> SolveResult:
    target = str(query.get("target") or "")
    unknown = str(query.get("unknown") or "q3")
    unknown_symbol = symbols.setdefault(unknown, sp.Symbol(unknown, real=True))
    ex, ey = _field_vector(schema, target, symbols)
    equations = []
    if sp.simplify(ex) != 0:
        equations.append(sp.Eq(ex, 0))
    if sp.simplify(ey) != 0:
        equations.append(sp.Eq(ey, 0))
    if not equations:
        expr = sp.Integer(0)
    else:
        solutions = sp.solve(equations, unknown_symbol, dict=True)
        if not solutions or unknown_symbol not in solutions[0]:
            raise SymbolicGeometryError(f"Could not solve zero-field condition for {unknown}.")
        expr = sp.simplify(solutions[0][unknown_symbol])
    unit = query.get("unit") or "C"
    # If the expression is numeric in SI and a display unit was requested, convert it.
    display_expr = expr
    if not expr.free_symbols and unit not in {None, "", "C"}:
        try:
            display_expr = sp.Float(convert(float(expr), "C", str(unit)))
        except Exception:
            display_expr = expr
    result = _make_result(display_expr, unit=unit, step=f"Solved symbolic zero-field vector equations for {unknown}.", domain=_result_domain(schema))
    return result


def _solve_midpoint_inverse_sqrt_relation(schema: dict[str, Any], query: dict[str, Any]) -> SolveResult:
    left = str(query.get("left") or "1/sqrt(E_M)")
    ea = sp.Symbol(str(query.get("left_endpoint_field") or "EA"), positive=True, real=True)
    eb = sp.Symbol(str(query.get("right_endpoint_field") or "EB"), positive=True, real=True)
    expr = sp.Rational(1, 2) * (1 / sp.sqrt(ea) + 1 / sp.sqrt(eb))
    display = f"{left} = {_render(expr)}"
    result = SolveResult(status="solved", domain=_result_domain(schema), answer=display)
    result.answer_meta = AnswerEnvelope.relation_answer(
        display=display,
        canonical=("sympy_relation", left, sp.sstr(sp.simplify(expr))),
        relation={"left": left, "right": sp.sstr(sp.simplify(expr))},
        unit=query.get("unit"),
        formula="point_charge_inverse_sqrt_midpoint_relation",
        source="symbolic_geometry_solver",
    )
    result.add_step("Symbolic geometry", "Used E=kq/r^2, so 1/sqrt(E) is proportional to distance from the charge; midpoint distances average linearly.")
    return result


def solve_schema(schema: dict[str, Any]) -> SolveResult:
    result = SolveResult(status="solve_failed", domain=_result_domain(schema))
    try:
        queries = schema.get("queries") or []
        if not queries:
            raise SymbolicGeometryError("symbolic_geometry schema needs a query.")
        query = queries[0]
        symbols = _collect_symbols(schema)
        qtype = str(query.get("type") or "electric_field")
        if qtype == "electric_field":
            return _solve_electric_field(schema, query, symbols)
        if qtype == "maximize_electric_field":
            return _solve_maximize_field(schema, query, symbols)
        if qtype == "zero_field_unknown_charge":
            return _solve_zero_field_unknown_charge(schema, query, symbols)
        if qtype == "midpoint_inverse_sqrt_field_relation":
            return _solve_midpoint_inverse_sqrt_relation(schema, query)
        raise SymbolicGeometryError(f"Unsupported symbolic_geometry query type: {qtype}.")
    except Exception as exc:
        result.error = str(exc)
        return result
