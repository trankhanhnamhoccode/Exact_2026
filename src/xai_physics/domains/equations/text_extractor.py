from __future__ import annotations

import re
from typing import Any

_SUPERSCRIPT_TRANS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
})

NUM = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:x|×|\*)\s*10\s*\^?\s*[-+]?\d+)?|10\s*\^?\s*[-+]?\d+)"
UNIT = r"(?:V/m|N/C|cm\^2|cm²|mm\^2|mm²|m\^2|m²|cm2|mm2|m2|μF|µF|uF|mF|nF|pF|F|μC|µC|uC|mC|nC|pC|C|cm|mm|m|V)"


def _norm(text: str) -> str:
    text = str(text).translate(_SUPERSCRIPT_TRANS)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("µ", "μ").replace("×", "x")
    return re.sub(r"\s+", " ", text).strip()


def _parse_number(raw: str) -> float:
    s = raw.strip().translate(_SUPERSCRIPT_TRANS)
    s = s.replace("−", "-").replace("×", "x").replace("*", "x")
    s = re.sub(r"\s+", "", s)
    m = re.fullmatch(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))x10\^?([-+]?\d+)", s, flags=re.I)
    if m:
        return float(m.group(1)) * (10.0 ** int(m.group(2)))
    m = re.fullmatch(r"([-+]?)10\^?([-+]?\d+)", s, flags=re.I)
    if m:
        return (-1.0 if m.group(1) == "-" else 1.0) * (10.0 ** int(m.group(2)))
    return float(s)


def _unit(raw: str) -> str:
    u = str(raw).strip().replace("µ", "μ")
    return u.replace("μF", "uF").replace("μC", "uC")


def _q(value: float, unit: str) -> dict[str, Any]:
    return {"value": value, "unit": _unit(unit)}


def _find(pattern: str, text: str) -> tuple[float, str] | None:
    m = re.search(pattern, text, flags=re.I)
    if not m:
        return None
    return _parse_number(m.group(1)), _unit(m.group(2))


def _radius(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:radius|R)\s*(?:=|is|of)?\s*({NUM})\s*(cm|mm|m)\b", text)


def _distance(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:distance|separation|d)\s*(?:between\s+the\s+plates\s+)?(?:=|is|of)?\s*({NUM})\s*(cm|mm|m)\b", text)


def _area(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:area(?: of each plate)?|plate area|A)\s*(?:=|is|of)?\s*({NUM})\s*(cm\^2|cm²|cm2|mm\^2|mm²|mm2|m\^2|m²|m2)\b", text)


def _field(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:electric field strength|field strength|electric field|E(?:_max)?)\D{{0,80}}?({NUM})\s*(V/m|N/C)\b", text)


def _capacitance(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:capacitance|C)\s*(?:=|is|of)?\s*({NUM})\s*(pF|nF|uF|μF|µF|mF|F)\b", text)


def _voltage(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:voltage|potential difference|U(?:AB)?)\s*(?:=|is|of)?\s*({NUM})\s*(V)\b", text)


def extract_equations_schema_from_text(problem: str) -> dict[str, Any] | None:
    text = _norm(problem)
    low = text.lower()

    # Parallel-plate capacitance with area/radius + separation.
    if "parallel" in low and "plate" in low and "capacitance" in low:
        r = _radius(text)
        a = _area(text)
        d = _distance(text)
        if d is not None and (r is not None or a is not None):
            objects: list[dict[str, Any]] = []
            if r is not None:
                objects.append({"id": "R1", "type": "radius", "role": "given", "value": str(r[0]), "unit": r[1]})
            else:
                objects.append({"id": "A1", "type": "area", "role": "given", "value": str(a[0]), "unit": a[1]})
            objects.append({"id": "d1", "type": "distance", "role": "given", "value": str(d[0]), "unit": d[1]})
            objects.append({"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "nF"})
            return {
                "domain": "equations",
                "objects": objects,
                "relations": [{"type": "formula", "name": "parallel_plate_capacitance", "objects": [obj["id"] for obj in objects]}],
                "constraints": [],
            }

    # Maximum charge before breakdown: Qmax = eps0 * eps_r * A * Emax.
    if "maximum charge" in low or "dielectric breakdown" in low or "breakdown" in low:
        r = _radius(text)
        a = _area(text)
        e = _field(text)
        if e is not None and (r is not None or a is not None):
            objects: list[dict[str, Any]] = []
            if r is not None:
                objects.append({"id": "R1", "type": "radius", "role": "given", "value": str(r[0]), "unit": r[1]})
            else:
                objects.append({"id": "A1", "type": "area", "role": "given", "value": str(a[0]), "unit": a[1]})
            objects.append({"id": "E_max", "type": "electric_field", "role": "given", "value": str(e[0]), "unit": e[1]})
            objects.append({"id": "Q_query", "type": "charge", "role": "query", "value": None, "unit": "uC"})
            return {
                "domain": "equations",
                "objects": objects,
                "relations": [{"type": "formula", "name": "parallel_plate_charge_from_field", "objects": [obj["id"] for obj in objects]}],
                "constraints": [],
            }

    # Two capacitors in series: voltage division.
    if "capacitor" in low and "series" in low and "voltage" in low:
        caps = re.findall(rf"\bC(\d+)\s*=\s*({NUM})\s*(pF|nF|uF|μF|µF|mF|F)\b", text, flags=re.I)
        u = _voltage(text)
        target_m = re.search(r"voltage\s+across\s+capacitor\s+C(\d+)", text, flags=re.I)
        if len(caps) >= 2 and u is not None:
            objects = []
            for idx, val, unit in caps[:2]:
                objects.append({"id": f"C{idx}", "type": "capacitance", "role": "given", "value": str(_parse_number(val)), "unit": _unit(unit)})
            objects.append({"id": "UAB", "type": "voltage", "role": "given", "value": str(u[0]), "unit": u[1]})
            target_idx = target_m.group(1) if target_m else caps[1][0]
            objects.append({"id": f"U{target_idx}_query", "type": "voltage", "role": "query", "value": None, "unit": "V"})
            return {
                "domain": "equations",
                "objects": objects,
                "relations": [{"type": "formula", "name": "capacitor_voltage_series", "objects": [obj["id"] for obj in objects]}],
                "constraints": [],
            }

    return None
