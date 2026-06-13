from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def _clean_symbol(raw: str) -> str:
    s = raw.strip().translate(_SUBSCRIPTS).translate(_SUPERSCRIPTS)
    s = s.replace("F0", "F0").replace("f0", "F0")
    if s.lower() in {"f0", "f₀"}:
        return "F0"
    return s


@dataclass(frozen=True)
class SymbolicExpr:
    coeff: Fraction = Fraction(1, 1)
    radicals: tuple[int, ...] = ()
    powers: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    @staticmethod
    def one() -> "SymbolicExpr":
        return SymbolicExpr()

    @staticmethod
    def symbol(name: str, power: int = 1) -> "SymbolicExpr":
        return SymbolicExpr(powers=((_clean_symbol(name), power),))._canonical()

    @staticmethod
    def number(value: int | Fraction) -> "SymbolicExpr":
        return SymbolicExpr(coeff=value if isinstance(value, Fraction) else Fraction(value, 1))

    @staticmethod
    def sqrt(radicand: int) -> "SymbolicExpr":
        return SymbolicExpr(radicals=(radicand,))._canonical()

    def _canonical(self) -> "SymbolicExpr":
        radical_counts: dict[int, int] = {}
        coeff = self.coeff
        for r in self.radicals:
            if r <= 0:
                raise ValueError("Radicands must be positive integers.")
            radical_counts[r] = radical_counts.get(r, 0) + 1
        radicals: list[int] = []
        for r, count in sorted(radical_counts.items()):
            coeff *= Fraction(r ** (count // 2), 1)
            if count % 2:
                radicals.append(r)

        powers: dict[str, int] = {}
        for sym, power in self.powers:
            sym = _clean_symbol(sym)
            powers[sym] = powers.get(sym, 0) + power
        clean_powers = tuple(sorted((sym, power) for sym, power in powers.items() if power != 0))
        return SymbolicExpr(coeff=coeff, radicals=tuple(radicals), powers=clean_powers)

    def __mul__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        return SymbolicExpr(
            coeff=self.coeff * other.coeff,
            radicals=(*self.radicals, *other.radicals),
            powers=(*self.powers, *other.powers),
        )._canonical()

    def __truediv__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        return SymbolicExpr(
            coeff=self.coeff / other.coeff,
            radicals=(*self.radicals, *other.radicals),
            powers=(*self.powers, *((sym, -power) for sym, power in other.powers)),
        )._canonical()

    def __neg__(self) -> "SymbolicExpr":
        return SymbolicExpr(coeff=-self.coeff, radicals=self.radicals, powers=self.powers)

    def render(self) -> str:
        expr = self._canonical()
        coeff = expr.coeff
        if coeff == 0:
            return "0"

        numerator: list[str] = []
        denominator: list[str] = []

        abs_coeff = abs(coeff)
        if abs_coeff != 1 or (not expr.radicals and not expr.powers):
            if abs_coeff.denominator == 1:
                numerator.append(str(abs_coeff.numerator))
            else:
                numerator.append(f"{abs_coeff.numerator}/{abs_coeff.denominator}")

        for r in expr.radicals:
            numerator.append(f"sqrt({r})")

        for sym, power in expr.powers:
            target = numerator if power > 0 else denominator
            p = abs(power)
            target.append(sym if p == 1 else f"{sym}^{p}")

        if not numerator:
            numerator.append("1")
        body = " × ".join(numerator)
        if denominator:
            body += " / " + " / ".join(denominator)
        return ("-" if coeff < 0 else "") + body

    def key(self) -> tuple:
        c = self._canonical()
        return (c.coeff, c.radicals, c.powers)


@dataclass(frozen=True)
class SymbolicRelation:
    left: str
    right: SymbolicExpr

    def render(self) -> str:
        return f"{self.left} = {self.right.render()}"

    def key(self) -> tuple:
        return (self.left.replace(" ", ""), self.right.key())


@dataclass(frozen=True)
class DirectionalAnswer:
    target: str

    def render(self) -> str:
        target = self.target.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
        return f"Hướng về phía {target}"

    def key(self) -> tuple:
        return ("direction", self.target.lower().translate(_SUBSCRIPTS))


def _preprocess(text: str) -> str:
    s = str(text).strip().translate(_SUBSCRIPTS).translate(_SUPERSCRIPTS)
    s = s.replace("F0", "F0").replace("f0", "F0")
    s = s.replace("E1", "E1").replace("e1", "E1").replace("E2", "E2").replace("e2", "E2")
    s = s.replace("Q", "q") if s.strip() == "Q" else s
    s = s.replace("×", "*").replace("·", "*")
    s = re.sub(r"\s+x\s+", "*", s, flags=re.I)
    s = s.replace("\\sqrt", "sqrt").replace("√", "sqrt")
    s = re.sub(r"sqrt\s*\{\s*(\d+)\s*\}", r"sqrt(\1)", s)
    s = re.sub(r"sqrt\s*\(\s*(\d+)\s*\)", r"sqrt(\1)", s)
    s = s.replace(" ", "")
    s = s.replace("**", "^")
    s = re.sub(r"\(?([+-]?\d+)\s*/\s*(\d+)\)?", r"frac(\1,\2)", s)
    s = re.sub(r"(?<=\d)(?=sqrt\()", "*", s)
    s = re.sub(r"(?<=\))(?=[A-Za-z])", "*", s)
    s = re.sub(r"(?<=[A-Za-z0-9])(?=sqrt\()", "*", s)
    s = re.sub(r"\*+", "*", s).strip("*")
    return s


def _parse_factor(token: str) -> SymbolicExpr | None:
    if not token:
        return SymbolicExpr.one()
    if token.startswith("(") and token.endswith(")"):
        return parse_symbolic_expr(token[1:-1])
    frac = re.fullmatch(r"frac\(([-+]?\d+),(\d+)\)", token) or re.fullmatch(r"([-+]?\d+)/(\d+)", token)
    if frac:
        return SymbolicExpr.number(Fraction(int(frac.group(1)), int(frac.group(2))))
    num = re.fullmatch(r"[-+]?\d+", token)
    if num:
        return SymbolicExpr.number(int(token))
    root = re.fullmatch(r"sqrt\(?([0-9]+)\)?", token)
    if root:
        return SymbolicExpr.sqrt(int(root.group(1)))
    var = re.fullmatch(r"([A-Za-z]+\d?)(?:\^([-+]?\d+))?", token)
    if var:
        name = var.group(1)
        power = int(var.group(2) or "1")
        if name == "f0":
            name = "F0"
        return SymbolicExpr.symbol(name, power)
    return None


def parse_symbolic_expr(text: str) -> SymbolicExpr | None:
    s = _preprocess(text)
    if not s:
        return None
    sign = 1
    if s.startswith("-"):
        sign = -1
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]

    parts = s.split("/")
    numerator = parts[0]
    denominators = parts[1:]
    expr = SymbolicExpr.number(sign)
    for token in filter(None, numerator.split("*")):
        factor = _parse_factor(token)
        if factor is None:
            return None
        expr = expr * factor
    for denom in denominators:
        for token in filter(None, denom.split("*")):
            factor = _parse_factor(token)
            if factor is None:
                return None
            expr = expr / factor
    return expr._canonical()


def parse_symbolic_answer(value: object) -> tuple | None:
    text = "" if value is None else str(value).strip()
    low = text.lower().translate(_SUBSCRIPTS)
    if "hướng" in low or "huong" in low or "toward" in low:
        m = re.search(r"q\s*([0-9]+)", low)
        if m:
            return DirectionalAnswer(f"q{m.group(1)}").key()
    if "=" in text:
        left, right = text.split("=", 1)
        left_norm = _preprocess(left).upper()
        expr = parse_symbolic_expr(right)
        if left_norm and expr is not None:
            return SymbolicRelation(left_norm, expr).key()
    expr = parse_symbolic_expr(text)
    if expr is not None:
        return ("expr", expr.key())
    return None
