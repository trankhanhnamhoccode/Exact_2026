from __future__ import annotations

from xai_physics.symbolic import SymbolicExpr, SymbolicRelation, parse_symbolic_answer


def test_symbolic_expr_canonicalizes_equivalent_sqrt_forms():
    expected = (SymbolicExpr.sqrt(2) * SymbolicExpr.symbol("F0")).key()
    assert parse_symbolic_answer(r"\sqrt{2} × F₀") == ("expr", expected)
    assert parse_symbolic_answer("F0 * sqrt(2)") == ("expr", expected)


def test_symbolic_expr_handles_products_and_denominators():
    expr = SymbolicExpr.number(2) * SymbolicExpr.sqrt(2) * SymbolicExpr.symbol("k") * SymbolicExpr.symbol("q") / SymbolicExpr.symbol("a", 2)
    assert parse_symbolic_answer("2 × sqrt(2) × k × q / a^2") == ("expr", expr.key())
    assert parse_symbolic_answer(expr.render()) == ("expr", expr.key())


def test_symbolic_relation_key():
    rel = SymbolicRelation("E1", SymbolicExpr.number(3) / SymbolicExpr.number(4) * SymbolicExpr.symbol("E2"))
    assert parse_symbolic_answer("E1 = (3/4)E2") == rel.key()
