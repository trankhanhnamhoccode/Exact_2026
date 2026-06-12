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
    "?F": 1e-6,
    "?C": 1e-6,
    "?J": 1e-6,
    "?N": 1e-6,
    "?m": 1e-6,
    "times": 1.0,
    "x": 1.0,
    "%": 1.0,
}


def normalize_unit(unit: str) -> str:
    # AUTO_MICRO_UNIT_NORMALIZATION
    unit = str(unit).strip()

    # Normalize common micro symbols and corrupted Windows/terminal variants.
    unit = unit.replace("?", "?")
    unit = unit.replace("?F", "uF")
    unit = unit.replace("?C", "uC")
    unit = unit.replace("?J", "uJ")
    unit = unit.replace("?N", "uN")
    unit = unit.replace("?m", "um")

    if unit in {"?F", "?F"}:
        return "uF"
    if unit in {"?C", "?C"}:
        return "uC"
    if unit in {"?J", "?J"}:
        return "uJ"
    if unit in {"?N", "?N"}:
        return "uN"
    if unit in {"?m", "?m"}:
        return "um"

    unit = str(unit).strip()

    # Normalize common micro symbols / mojibake.
    unit = unit.replace("µ", "μ")

    # Windows / terminal encoding may corrupt μ into ?.
    # Treat exact ?F, ?C, ?J, ?N, ?m as micro-units.
    if unit in {"?F", "？F"}:
        return "uF"
    if unit in {"?C", "？C"}:
        return "uC"
    if unit in {"?J", "？J"}:
        return "uJ"
    if unit in {"?N", "？N"}:
        return "uN"
    if unit in {"?m", "？m"}:
        return "um"

    # Normalize micro symbol to ASCII aliases already supported by UNIT_TO_SI.
    unit = unit.replace("μF", "uF")
    unit = unit.replace("μC", "uC")
    unit = unit.replace("μJ", "uJ")
    unit = unit.replace("μN", "uN")
    unit = unit.replace("μm", "um")

    unit = unit.replace("Ohm", "ohm")

    return unit


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
