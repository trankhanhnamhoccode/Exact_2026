from __future__ import annotations

import re
from typing import Any

_SUPERSCRIPT_TRANS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
})

NUM = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:x|×|\*)\s*10\s*\^?\s*[-+]?\d+)?|10\s*\^?\s*[-+]?\d+)"
UNIT = r"(?:rad/s|V/m|N/C|cm\^2|cm²|mm\^2|mm²|m\^2|m²|cm2|mm2|m2|kΩ|kohm|MΩ|Mohm|Ω|ohm|μF|µF|uF|mF|nF|pF|F|μC|µC|uC|mC|nC|pC|C|μH|µH|uH|mH|H|mA|A|kHz|Hz|ms|s|mJ|μJ|µJ|uJ|J|mW|W|kg|g|cm|mm|m|°C|Celsius|V)"


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
    # Only treat bare 10 as scientific notation when there is an explicit
    # exponent marker/sign.  Without this guard, ordinary values like 100 Hz
    # were parsed as 10^0 = 1.
    m = re.fullmatch(r"([-+]?)10(?:\^([-+]?\d+)|([-+]\d+))", s, flags=re.I)
    if m:
        exponent = m.group(2) if m.group(2) is not None else m.group(3)
        return (-1.0 if m.group(1) == "-" else 1.0) * (10.0 ** int(exponent))
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





def _voltage_any(text: str) -> tuple[float, str] | None:
    found = _voltage(text)
    if found is not None:
        return found
    for pattern in (
        rf"(?:source|voltage source|battery|supply)\D{{0,20}}?({NUM})\s*(V)\b",
        rf"({NUM})\s*(V)\b\s*(?:source|battery|supply)",
        rf"\bU\s*=\s*({NUM})\s*(V)\b",
    ):
        found = _find(pattern, text)
        if found is not None:
            return found
    return None


def _current_any(text: str) -> tuple[float, str] | None:
    found = _current(text)
    if found is not None:
        return found
    for pattern in (
        rf"(?:supplies|supply|delivers|total current|current)\D{{0,40}}?({NUM})\s*(mA|A)\b",
        rf"\bI\s*=\s*({NUM})\s*(mA|A)\b",
    ):
        found = _find(pattern, text)
        if found is not None:
            return found
    return None


def _power_values(text: str) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for m in re.finditer(rf"({NUM})\s*(mW|W)\b", text, flags=re.I):
        out.append((_parse_number(m.group(1)), _unit(m.group(2))))
    return out


def _lamp_count(low: str) -> int:
    word_counts = {"two": 2, "three": 3, "four": 4, "five": 5}
    for word, value in word_counts.items():
        if re.search(rf"\b{word}\s+(?:identical\s+)?(?:lamps|bulbs|resistors|branches)\b", low):
            return value
    m = re.search(r"\b(\d+)\s+(?:identical\s+)?(?:lamps|bulbs|resistors|branches)\b", low)
    if m:
        return max(1, int(m.group(1)))
    return 2


def _equal_lamp_resistance(text: str) -> tuple[float, str] | None:
    for pattern in (
        rf"(?:each\s+(?:lamp|bulb|branch|resistor)\s+(?:has|has a|with)\s+(?:a\s+)?resistance\s*(?:R\s*)?(?:=|is|of)?\s*)({NUM})\s*(Ω|ohm|kohm|kΩ|Mohm|MΩ)\b",
        rf"(?:resistance\s*(?:R\s*)?(?:=|is|of)?\s*)({NUM})\s*(Ω|ohm|kohm|kΩ|Mohm|MΩ)\b",
    ):
        found = _find(pattern, text)
        if found is not None:
            return found
    return None


def _measurement_uncertainty_pair(text: str) -> tuple[float, float, str] | None:
    m = re.search(rf"({NUM})\s*(?:±|\+/-|\+-)\s*({NUM})\s*({UNIT})\b", text, flags=re.I)
    if not m:
        return None
    return _parse_number(m.group(1)), _parse_number(m.group(2)), _unit(m.group(3))


def _measurement_quantity_type(low: str) -> str:
    for name in ("voltage", "current", "resistance", "temperature", "length", "mass", "time", "force", "power"):
        if name in low:
            return name
    return "measured_value"


def _frequency_any(text: str) -> tuple[float, str] | None:
    found = _frequency(text)
    if found is not None:
        return found
    for pattern in (
        rf"(?:resonate|resonates|resonance|resonant|at)\D{{0,40}}?({NUM})\s*(kHz|Hz)\b",
        rf"({NUM})\s*(kHz|Hz)\b\D{{0,40}}?(?:resonate|resonance|resonant)",
    ):
        found = _find(pattern, text)
        if found is not None:
            return found
    return None


def _capacitance_any(text: str) -> tuple[float, str] | None:
    found = _capacitance(text)
    if found is not None:
        return found
    return _find(rf"({NUM})\s*(pF|nF|uF|μF|µF|mF|F)\s*(?:capacitor|capacitance)", text)


def _inductance_any(text: str) -> tuple[float, str] | None:
    found = _inductance(text)
    if found is not None:
        return found
    return _find(rf"({NUM})\s*(uH|μH|µH|mH|H)\s*(?:inductor|inductance)", text)


def _asks_capacitance(low: str) -> bool:
    return "what capacitance" in low or "capacitance c" in low or "capacitor value" in low or "what value of c" in low or "what c" in low


def _asks_inductance(low: str) -> bool:
    return "what inductance" in low or "inductance l" in low or "what inductor" in low or "what l" in low or "inductor l" in low or "l is required" in low or "l is needed" in low


def _asks_resistance(low: str) -> bool:
    return "resistance" in low or "pure resistance" in low or re.search(r"\bcalculate\s+r\b|\bdetermine\s+r\b|\bwhat\s+is\s+r\b", low) is not None


def _asks_impedance(low: str) -> bool:
    return "impedance" in low or re.search(r"\bwhat\s+is\s+z\b|\bcalculate\s+z\b", low) is not None


def extract_equations_schema_from_text(problem: str) -> dict[str, Any] | None:
    text = _norm(problem)
    low = text.lower()

    # Deterministic measurement shortcut: "x = value ± uncertainty unit" -> percent relative error.
    if ("relative error" in low or "relative uncertainty" in low or "percentage" in low or "percent" in low) and ("±" in text or "+/-" in text or "+-" in text):
        pair = _measurement_uncertainty_pair(text)
        if pair is not None:
            value, uncertainty, unit = pair
            qtype = _measurement_quantity_type(low)
            objects = [
                {"id": "measured1", "type": qtype if qtype != "measured_value" else "measured_value", "role": "given", "value": str(value), "unit": unit},
                {"id": "err1", "type": f"{qtype}_uncertainty" if qtype != "measured_value" else "absolute_error", "role": "given", "value": str(uncertainty), "unit": unit},
                {"id": "percent_query", "type": "percent_error", "role": "query", "value": None, "unit": "%"},
            ]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "percentage_relative_error", "objects": [obj["id"] for obj in objects]}], "constraints": []}

    # Deterministic parallel lamp shortcut for branch currents + total current in one answer.
    if ("parallel" in low and ("lamp" in low or "bulb" in low or "branch" in low)) and "current" in low:
        u = _voltage_any(text)
        r = _equal_lamp_resistance(text)
        if u is not None and r is not None and ("each" in low or "total" in low):
            n = _lamp_count(low)
            objects: list[dict[str, Any]] = [
                {"id": "U1", "type": "voltage", "role": "given", "value": str(u[0]), "unit": u[1]},
            ]
            for idx in range(1, n + 1):
                objects.append({"id": f"R{idx}", "type": "resistance", "role": "given", "value": str(r[0]), "unit": r[1]})
            for idx in range(1, n + 1):
                objects.append({"id": f"I{idx}_query", "type": "current", "role": "query", "value": None, "unit": "A", "symbol": f"I{idx}"})
            objects.append({"id": "Itotal_query", "type": "total_current", "role": "query", "value": None, "unit": "A"})
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "parallel_all_currents", "objects": [obj["id"] for obj in objects]}], "constraints": []}

    # Direct total power P = UI when total source voltage and total current are stated.
    if "power" in low and ("source" in low or "supplies" in low or "total" in low):
        u = _voltage_any(text)
        i = _current_any(text)
        if u is not None and i is not None:
            objects = [
                {"id": "U1", "type": "voltage", "role": "given", "value": str(u[0]), "unit": u[1]},
                {"id": "I1", "type": "current", "role": "given", "value": str(i[0]), "unit": i[1]},
                {"id": "P_query", "type": "power", "role": "query", "value": None, "unit": "W"},
            ]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "power_voltage_current", "objects": [obj["id"] for obj in objects]}], "constraints": []}

    # Total power from listed lamp powers: P_total = sum(P_i).
    if "power" in low and "total" in low and ("lamp" in low or "bulb" in low):
        powers = _power_values(text)
        if len(powers) >= 2:
            objects = [
                {"id": f"P{idx}", "type": "power", "role": "given", "value": str(value), "unit": unit}
                for idx, (value, unit) in enumerate(powers, 1)
            ]
            objects.append({"id": "Ptotal_query", "type": "total_power", "role": "query", "value": None, "unit": "W"})
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "total_power_sum", "objects": [obj["id"] for obj in objects]}], "constraints": []}



    # RLC resonance identity: at resonance, Z = R and cos(phi)=1.
    if "reson" in low or "z = r" in low or "z=r" in low or "φ = 0" in low or "phi = 0" in low:
        if "power factor" in low or "cosφ" in low or "cos phi" in low or "cosphi" in low or "cos(phi" in low:
            objects = [{"id": "cos_phi_query", "type": "power_factor", "role": "query", "value": None, "unit": ""}]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "power_factor_at_resonance", "objects": ["cos_phi_query"]}], "constraints": ["At resonance, phi=0 so cos(phi)=1."]}

        z = _impedance(text)
        r = _resistance(text)
        if z is not None and _asks_resistance(low):
            objects = [
                {"id": "Z1", "type": "impedance", "role": "given", "value": str(z[0]), "unit": z[1]},
                {"id": "R_query", "type": "resistance", "role": "query", "value": None, "unit": "ohm"},
            ]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "rlc_resonance_impedance_resistance", "objects": [obj["id"] for obj in objects]}], "constraints": ["At resonance, Z=R."]}
        if r is not None and _asks_impedance(low):
            objects = [
                {"id": "R1", "type": "resistance", "role": "given", "value": str(r[0]), "unit": r[1]},
                {"id": "Z_query", "type": "impedance", "role": "query", "value": None, "unit": "ohm"},
            ]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "rlc_resonance_impedance_resistance", "objects": [obj["id"] for obj in objects]}], "constraints": ["At resonance, Z=R."]}

    # LC/RLC inverse resonance from text, avoiding LLM schemas that lose f=... as f_query=null.
    if "reson" in low or "f0" in low or "natural frequency" in low:
        l = _inductance_any(text)
        c = _capacitance_any(text)
        f = _frequency_any(text)
        if l is not None and f is not None and _asks_capacitance(low):
            objects = [
                {"id": "L1", "type": "inductance", "role": "given", "value": str(l[0]), "unit": l[1]},
                {"id": "f1", "type": "frequency", "role": "given", "value": str(f[0]), "unit": f[1]},
                {"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "uF"},
            ]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "lc_resonance_frequency", "objects": [obj["id"] for obj in objects]}], "constraints": []}
        if c is not None and f is not None and _asks_inductance(low):
            objects = [
                {"id": "C1", "type": "capacitance", "role": "given", "value": str(c[0]), "unit": c[1]},
                {"id": "f1", "type": "frequency", "role": "given", "value": str(f[0]), "unit": f[1]},
                {"id": "L_query", "type": "inductance", "role": "query", "value": None, "unit": "H"},
            ]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "lc_resonance_frequency", "objects": [obj["id"] for obj in objects]}], "constraints": []}
        if l is not None and c is not None and ("calculate f0" in low or "calculate f_0" in low or "frequency" in low):
            objects = [
                {"id": "L1", "type": "inductance", "role": "given", "value": str(l[0]), "unit": l[1]},
                {"id": "C1", "type": "capacitance", "role": "given", "value": str(c[0]), "unit": c[1]},
                {"id": "f_query", "type": "frequency", "role": "query", "value": None, "unit": "Hz"},
            ]
            return {"domain": "equations", "objects": objects, "relations": [{"type": "formula", "name": "lc_resonance_frequency", "objects": [obj["id"] for obj in objects]}], "constraints": []}

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
