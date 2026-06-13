from __future__ import annotations

import re
from typing import Any

_SUPERSCRIPT_TRANS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
})

NUM = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:x|×|\*)\s*10\s*\^?\s*[-+]?\d+)?|10\s*\^?\s*[-+]?\d+)"
UNIT = r"(?:rad/s|V/m|N/C|cm\^2|cm²|mm\^2|mm²|m\^2|m²|cm2|mm2|m2|kΩ|kohm|MΩ|Mohm|Ω|ohm|μF|µF|uF|mF|nF|pF|F|μC|µC|uC|mC|nC|pC|C|μH|µH|uH|mH|H|mA|A|kHz|Hz|ms|s|mJ|μJ|µJ|uJ|J|mW|W|cm|mm|m|V)"


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
    return (
        u.replace("μF", "uF")
        .replace("μC", "uC")
        .replace("μH", "uH")
        .replace("μJ", "uJ")
        .replace("Ω", "Ω")
    )


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


def _inductance(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:inductance|coil has an inductance|inductance L|\bL)\s*(?:=|is|of|has)?\s*({NUM})\s*(uH|μH|µH|mH|H)\b", text)


def _current(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:maximum current|current through the coil|current flows through it|current|\bI(?:0|max)?)\s*(?:=|is|of|through it|flows through it)?\s*({NUM})\s*(mA|A)\b", text)


def _energy(text: str) -> tuple[float, str] | None:
    found = _find(rf"(?:maximum magnetic energy|magnetic field energy|magnetic energy|total energy|energy)\s*(?:=|is|of|stores|stored|when)?\s*({NUM})\s*(uJ|μJ|µJ|mJ|J)\b", text)
    if found is not None:
        return found
    return _find(rf"(?:store|stores|stored)\s*({NUM})\s*(uJ|μJ|µJ|mJ|J)\s*(?:of\s+)?(?:magnetic\s+)?energy", text)


def _frequency(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:frequency|\bf)\s*(?:=|is|of)?\s*({NUM})\s*(kHz|Hz)\b", text)


def _angular_frequency(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:angular frequency|omega|ω)\s*(?:=|is|of)?\s*({NUM})\s*(rad/s)\b", text)


def _period(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:period|\bT)\s*(?:=|is|of)?\s*({NUM})\s*(ms|s)\b", text)


def _resistance(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:resistance|\bR)\s*(?:=|is|of)?\s*({NUM})\s*(Ω|ohm|kohm|kΩ|Mohm|MΩ)\b", text)


def _impedance(text: str) -> tuple[float, str] | None:
    return _find(rf"(?:impedance|\bZ)\s*(?:=|is|of)?\s*({NUM})\s*(Ω|ohm|kohm|kΩ|Mohm|MΩ)\b", text)



def extract_equations_schema_from_text(problem: str) -> dict[str, Any] | None:
    text = _norm(problem)
    low = text.lower()

    # Inductor scalar energy: W = 1/2*L*I^2, with inverse forms.
    if ("inductor" in low or "coil" in low) and ("magnetic" in low or "energy" in low):
        l = _inductance(text)
        i = _current(text)
        w = _energy(text)
        if l is not None and i is not None and ("what" in low or "calculate" in low or "find" in low) and "energy" in low:
            objects = [
                {"id": "L1", "type": "inductance", "role": "given", "value": str(l[0]), "unit": l[1]},
                {"id": "I1", "type": "current", "role": "given", "value": str(i[0]), "unit": i[1]},
                {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "J"},
            ]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "inductor_energy", "objects": [obj["id"] for obj in objects]}], "constraints": []}
        if l is not None and w is not None and ("current" in low or "maximum current" in low):
            objects = [
                {"id": "L1", "type": "inductance", "role": "given", "value": str(l[0]), "unit": l[1]},
                {"id": "W1", "type": "energy", "role": "given", "value": str(w[0]), "unit": w[1]},
                {"id": "I_query", "type": "current", "role": "query", "value": None, "unit": "A"},
            ]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "inductor_energy", "objects": [obj["id"] for obj in objects]}], "constraints": []}
        if i is not None and w is not None and "inductance" in low:
            objects = [
                {"id": "I1", "type": "current", "role": "given", "value": str(i[0]), "unit": i[1]},
                {"id": "W1", "type": "energy", "role": "given", "value": str(w[0]), "unit": w[1]},
                {"id": "L_query", "type": "inductance", "role": "query", "value": None, "unit": "H"},
            ]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "inductor_energy", "objects": [obj["id"] for obj in objects]}], "constraints": []}

    # LC natural oscillation scalar extraction.
    if ("lc" in low or "oscillat" in low) and ("natural" in low or "angular frequency" in low or "period" in low):
        l = _inductance(text)
        c = _capacitance(text)
        if l is not None and c is not None:
            if "angular frequency" in low or "omega" in low or "ω" in low:
                qid, qtype, unit, formula = "omega_query", "angular_frequency", "rad/s", "lc_resonance_angular_frequency"
            elif "period" in low:
                qid, qtype, unit, formula = "T_query", "period", "s", "lc_natural_period"
            else:
                qid, qtype, unit, formula = "f_query", "frequency", "Hz", "lc_resonance_frequency"
            objects = [
                {"id": "L1", "type": "inductance", "role": "given", "value": str(l[0]), "unit": l[1]},
                {"id": "C1", "type": "capacitance", "role": "given", "value": str(c[0]), "unit": c[1]},
                {"id": qid, "type": qtype, "role": "query", "value": None, "unit": unit},
            ]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": formula, "objects": [obj["id"] for obj in objects]}], "constraints": []}

    # RLC power from voltage, impedance, and resistance: P=(U/Z)^2*R.
    if ("power" in low or "power consumed" in low) and ("impedance" in low or " z " in f" {low} "):
        u = _voltage(text)
        z = _impedance(text)
        r = _resistance(text)
        if u is not None and z is not None and r is not None:
            objects = [
                {"id": "U1", "type": "voltage", "role": "given", "value": str(u[0]), "unit": u[1]},
                {"id": "Z1", "type": "impedance", "role": "given", "value": str(z[0]), "unit": z[1]},
                {"id": "R1", "type": "resistance", "role": "given", "value": str(r[0]), "unit": r[1]},
                {"id": "P_query", "type": "power", "role": "query", "value": None, "unit": "W"},
            ]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "rlc_power_voltage_impedance_resistance", "objects": [obj["id"] for obj in objects]}], "constraints": []}

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
