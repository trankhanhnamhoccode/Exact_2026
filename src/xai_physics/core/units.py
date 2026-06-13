from __future__ import annotations


UNIT_TO_SI = {
    # charge
    "C": 1.0,
    "mC": 1e-3,
    "uC": 1e-6,
    "μC": 1e-6,
    "nC": 1e-9,
    "pC": 1e-12,

    # voltage
    "V": 1.0,
    "mV": 1e-3,
    "kV": 1e3,

    # capacitance
    "F": 1.0,
    "mF": 1e-3,
    "uF": 1e-6,
    "μF": 1e-6,
    "nF": 1e-9,
    "pF": 1e-12,

    # resistance
    "ohm": 1.0,
    "Ω": 1.0,
    "kohm": 1e3,
    "kΩ": 1e3,
    "Mohm": 1e6,
    "MΩ": 1e6,

    # length
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "μm": 1e-6,
    "nm": 1e-9,

    # area
    "m^2": 1.0,
    "m²": 1.0,
    "cm^2": 1e-4,
    "cm²": 1e-4,
    "mm^2": 1e-6,
    "mm²": 1e-6,

    # energy
    "J": 1.0,
    "mJ": 1e-3,
    "uJ": 1e-6,
    "μJ": 1e-6,
    "nJ": 1e-9,
    "pJ": 1e-12,

    # force
    "N": 1.0,
    "mN": 1e-3,
    "uN": 1e-6,
    "μN": 1e-6,

    # electric field
    "V/m": 1.0,
    "N/C": 1.0,
    "kV/m": 1e3,

    # angle / dimensionless helpers
    "degree": 1.0,
    "degrees": 1.0,
    "deg": 1.0,
    "?F": 1e-6,
    "?C": 1e-6,
    "?J": 1e-6,
    "?N": 1e-6,
    "?m": 1e-6,
    "times": 1.0,
    "x": 1.0,
    "%": 1.0,
}


def normalize_unit(unit: str | None) -> str:
    if unit is None:
        return ""

    unit = str(unit).strip()
    if not unit or unit.lower() in {"none", "null", "nan"}:
        return ""

    unit = unit.replace("µ", "μ")
    unit = unit.replace("Ω", "Ω")
    unit = unit.replace("^2", "²")
    unit = unit.replace("^3", "³")
    unit = unit.replace("Ohm", "ohm")
    unit = unit.replace("ohms", "ohm")

    # Normalize spaces around slash: "N / C" -> "N/C".
    unit = unit.replace(" / ", "/").replace(" /", "/").replace("/ ", "/")

    mojibake = {
        "?F": "uF", "?C": "uC", "?J": "uJ", "?N": "uN", "?m": "um",
        "？F": "uF", "？C": "uC", "？J": "uJ", "？N": "uN", "？m": "um",
    }
    if unit in mojibake:
        return mojibake[unit]

    # Normalize micro symbol to ASCII aliases already supported by UNIT_TO_SI.
    unit = unit.replace("μF", "uF")
    unit = unit.replace("μC", "uC")
    unit = unit.replace("μJ", "uJ")
    unit = unit.replace("μN", "uN")
    unit = unit.replace("μm", "um")

    aliases = {
        # bad placeholders
        "unitless": "",
        "dimensionless": "",
        "si": "",

        # charge aliases, but intentionally do NOT map bare "c" here.
        # In electrostatics geometry Qwen often writes "c" for "cm".
        "coulomb": "C",
        "coulombs": "C",
        "mc": "mC",
        "millicoulomb": "mC",
        "millicoulombs": "mC",
        "uc": "uC",
        "microcoulomb": "uC",
        "microcoulombs": "uC",
        "nc": "nC",
        "nanocoulomb": "nC",
        "nanocoulombs": "nC",
        "pc": "pC",

        # force aliases, but intentionally do NOT map bare "n" globally.
        "newton": "N",
        "newtons": "N",
        "mn": "mN",
        "un": "uN",

        # field
        "v/m": "V/m",
        "volt/meter": "V/m",
        "volts/meter": "V/m",
        "n/c": "N/C",
        "newton/coulomb": "N/C",
        "newtons/coulomb": "N/C",

        # voltage
        "v": "V",
        "mv": "mV",
        "kv": "kV",

        # capacitance
        "f": "F",
        "farad": "F",
        "farads": "F",
        "mf": "mF",
        "uf": "uF",
        "nf": "nF",
        "pf": "pF",

        # length
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "centimeter": "cm",
        "centimeters": "cm",
        "centimetre": "cm",
        "centimetres": "cm",
        "millimeter": "mm",
        "millimeters": "mm",
        "millimetre": "mm",
        "millimetres": "mm",

        # area
        "m2": "m^2",
        "cm2": "cm^2",
        "mm2": "mm^2",

        # energy
        "j": "J",
        "joule": "J",
        "joules": "J",
        "mj": "mJ",
        "uj": "uJ",
        "nj": "nJ",
        "pj": "pJ",

        # resistance
        "ω": "Ω",
        "kohm": "kohm",
        "kω": "kΩ",
        "mohm": "Mohm",
        "mω": "MΩ",
    }

    return aliases.get(unit.lower(), unit)


def to_si(value: float, unit: str) -> float:
    unit = normalize_unit(unit)
    if unit not in UNIT_TO_SI:
        raise ValueError(f"Unsupported unit: {unit}")
    return value * UNIT_TO_SI[unit]


def convert(value: float, from_unit: str, to_unit: str) -> float:
    from_unit = normalize_unit(from_unit)
    to_unit = normalize_unit(to_unit)

    value_si = to_si(value, from_unit)

    if to_unit not in UNIT_TO_SI:
        raise ValueError(f"Unsupported unit: {to_unit}")

    return value_si / UNIT_TO_SI[to_unit]
