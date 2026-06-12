from __future__ import annotations

from typing import Any

from .types import FormulaDoc


def _schema(formula_name: str, objects: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "domain": "equations",
        "objects": objects,
        "relations": [
            {
                "type": "formula",
                "name": formula_name,
                "objects": [obj["id"] for obj in objects],
            }
        ],
        "constraints": [],
    }


FORMULA_DOCS: list[FormulaDoc] = [
    FormulaDoc(
        id="capacitor_energy_voltage",
        name="Capacitor energy from capacitance and voltage",
        equation="W = 1/2 C U^2",
        description="Use for scalar capacitor energy when capacitance and voltage are known.",
        quantity_types=["energy", "capacitance", "voltage"],
        query_types=["energy", "capacitance", "voltage"],
        tags=["capacitor", "energy", "voltage", "capacitance"],
        keywords=["stored energy", "energy stored", "capacitor energy", "C and U", "capacitance voltage"],
        schema_template=_schema(
            "capacitor_energy_voltage",
            [
                {"id": "C1", "type": "capacitance", "role": "given", "value": "<number>", "unit": "<F|uF|nF|pF>"},
                {"id": "U1", "type": "voltage", "role": "given", "value": "<number>", "unit": "<V|kV|mV>"},
                {"id": "W_query", "type": "energy", "role": "query", "value": None, "unit": "<J|mJ|uJ>"},
            ],
        ),
    ),
    FormulaDoc(
        id="capacitor_charge_voltage",
        name="Capacitor charge-voltage relation",
        equation="Q = C U",
        description="Use for direct scalar relation among charge, capacitance, and voltage.",
        quantity_types=["charge", "capacitance", "voltage"],
        query_types=["charge", "capacitance", "voltage"],
        tags=["capacitor", "charge", "voltage", "capacitance"],
        keywords=["charge on capacitor", "potential difference", "Q equals C U", "find charge", "find voltage"],
        schema_template=_schema(
            "capacitor_charge_voltage",
            [
                {"id": "Q_query", "type": "charge", "role": "query", "value": None, "unit": "<C|mC|uC|nC>"},
                {"id": "C1", "type": "capacitance", "role": "given", "value": "<number>", "unit": "<F|uF|nF|pF>"},
                {"id": "U1", "type": "voltage", "role": "given", "value": "<number>", "unit": "<V|kV|mV>"},
            ],
        ),
    ),
    FormulaDoc(
        id="parallel_plate_capacitance",
        name="Parallel-plate capacitance",
        equation="C = eps0 eps_r A / d",
        description="Use for capacitance of a parallel-plate capacitor from area, separation, and dielectric constant.",
        quantity_types=["capacitance", "area", "distance", "relative_permittivity"],
        query_types=["capacitance", "relative_permittivity", "area", "distance"],
        tags=["capacitor", "parallel_plate", "capacitance", "area", "distance", "dielectric"],
        keywords=["parallel plate", "plate area", "plate separation", "air capacitor", "dielectric constant"],
        schema_template=_schema(
            "parallel_plate_capacitance",
            [
                {"id": "A1", "type": "area", "role": "given", "value": "<number>", "unit": "<m2|cm2|mm2>"},
                {"id": "d1", "type": "distance", "role": "given", "value": "<number>", "unit": "<m|cm|mm>"},
                {"id": "C_query", "type": "capacitance", "role": "query", "value": None, "unit": "<F|uF|nF|pF>"},
            ],
        ),
    ),
    FormulaDoc(
        id="series_capacitance_unknown",
        name="Two capacitors in series, unknown capacitance",
        equation="1/Ceq = 1/C1 + 1/C2",
        description="Use when equivalent capacitance of two series capacitors is known and one capacitance is unknown.",
        quantity_types=["capacitance"],
        query_types=["capacitance"],
        tags=["capacitor", "series", "equivalent_capacitance", "algebra"],
        keywords=["series capacitors", "connected in series", "equivalent capacitance", "unknown capacitance"],
        schema_template=_schema(
            "series_capacitance_unknown",
            [
                {"id": "Ceq", "type": "capacitance", "role": "given", "value": "<number>", "unit": "<F|uF|nF|pF>", "symbol": "Ceq"},
                {"id": "C1", "type": "capacitance", "role": "given", "value": "<number>", "unit": "<F|uF|nF|pF>", "symbol": "C1"},
                {"id": "C2_query", "type": "capacitance", "role": "query", "value": None, "unit": "<F|uF|nF|pF>", "symbol": "C2"},
            ],
        ),
    ),
    FormulaDoc(
        id="lc_resonance_frequency",
        name="LC resonance frequency",
        equation="f = 1/(2*pi*sqrt(L*C))",
        description="Use for natural/resonant frequency of an LC circuit from inductance and capacitance.",
        quantity_types=["frequency", "inductance", "capacitance"],
        query_types=["frequency", "inductance", "capacitance"],
        tags=["lc", "resonance", "frequency", "inductance", "capacitance"],
        keywords=["resonance frequency", "resonant frequency", "natural frequency", "LC circuit", "oscillating circuit"],
        schema_template=_schema(
            "lc_resonance_frequency",
            [
                {"id": "L1", "type": "inductance", "role": "given", "value": "<number>", "unit": "<H|mH|uH>"},
                {"id": "C1", "type": "capacitance", "role": "given", "value": "<number>", "unit": "<F|uF|nF|pF>"},
                {"id": "f_query", "type": "frequency", "role": "query", "value": None, "unit": "<Hz|kHz>"},
            ],
        ),
    ),
    FormulaDoc(
        id="ohm_law",
        name="Ohm law",
        equation="U = I R",
        description="Use for scalar voltage-current-resistance relation, including resonance when impedance equals resistance.",
        quantity_types=["voltage", "current", "resistance"],
        query_types=["voltage", "current", "resistance"],
        tags=["circuit", "ohm", "voltage", "current", "resistance"],
        keywords=["ohm law", "current through resistor", "voltage resistance current", "at resonance impedance equals resistance"],
        schema_template=_schema(
            "ohm_law",
            [
                {"id": "U1", "type": "voltage", "role": "given", "value": "<number>", "unit": "<V|kV|mV>"},
                {"id": "R1", "type": "resistance", "role": "given", "value": "<number>", "unit": "<?|ohm>"},
                {"id": "I_query", "type": "current", "role": "query", "value": None, "unit": "<A|mA>"},
            ],
        ),
    ),
    FormulaDoc(
        id="quality_factor",
        name="Series RLC quality factor",
        equation="Q = sqrt(L/C)/R",
        description="Use for quality factor of a series RLC circuit from L, C, and R.",
        quantity_types=["quality_factor", "inductance", "capacitance", "resistance"],
        query_types=["quality_factor"],
        tags=["rlc", "quality_factor", "inductance", "capacitance", "resistance"],
        keywords=["quality factor", "Q factor", "series RLC", "sqrt L over C divided by R"],
        schema_template=_schema(
            "quality_factor",
            [
                {"id": "L1", "type": "inductance", "role": "given", "value": "<number>", "unit": "<H|mH|uH>"},
                {"id": "C1", "type": "capacitance", "role": "given", "value": "<number>", "unit": "<F|uF|nF|pF>"},
                {"id": "R1", "type": "resistance", "role": "given", "value": "<number>", "unit": "<?|ohm>"},
                {"id": "Q_query", "type": "quality_factor", "role": "query", "value": None, "unit": "-"},
            ],
        ),
    ),
    FormulaDoc(
        id="solenoid_magnetic_field",
        name="Magnetic field inside a long solenoid",
        equation="B = mu0*N*I/l = mu0*n*I",
        description="Use for magnetic field in a long solenoid from turn count or turn density and current.",
        quantity_types=["magnetic_field", "turn_count", "turn_density", "current", "length"],
        query_types=["magnetic_field", "current"],
        tags=["solenoid", "magnetic_field", "turns", "current", "length"],
        keywords=["solenoid", "magnetic field", "number of turns", "turn density", "inside a solenoid"],
        schema_template=_schema(
            "solenoid_magnetic_field",
            [
                {"id": "N1", "type": "turn_count", "role": "given", "value": "<number>", "unit": "turns"},
                {"id": "I1", "type": "current", "role": "given", "value": "<number>", "unit": "<A|mA>"},
                {"id": "l1", "type": "length", "role": "given", "value": "<number>", "unit": "<m|cm>"},
                {"id": "B_query", "type": "magnetic_field", "role": "query", "value": None, "unit": "<T|mT>"},
            ],
        ),
    ),
    FormulaDoc(
        id="point_charge_electric_field",
        name="Electric field of a point charge",
        equation="E = k|q|/(eps_r*r^2)",
        description="Use for scalar electric field magnitude due to one point charge at a distance.",
        quantity_types=["electric_field", "charge", "distance", "relative_permittivity"],
        query_types=["electric_field"],
        tags=["electric_field", "point_charge", "charge", "distance"],
        keywords=["electric field due to point charge", "field at distance", "point charge electric field"],
        schema_template=_schema(
            "point_charge_electric_field",
            [
                {"id": "q1", "type": "charge", "role": "given", "value": "<number>", "unit": "<C|mC|uC|nC>"},
                {"id": "r1", "type": "distance", "role": "given", "value": "<number>", "unit": "<m|cm|mm>"},
                {"id": "E_query", "type": "electric_field", "role": "query", "value": None, "unit": "<V/m|N/C>"},
            ],
        ),
    ),
    FormulaDoc(
        id="percentage_relative_error",
        name="Percentage relative error",
        equation="percent_error = absolute_error/measured_value * 100%",
        description="Use when the problem asks for percentage relative error from absolute error and measured value.",
        quantity_types=["percent_error", "absolute_error", "measured_value"],
        query_types=["percent_error"],
        tags=["measurement_error", "relative_error", "percentage"],
        keywords=["percentage error", "relative error", "absolute error", "measured value"],
        schema_template=_schema(
            "percentage_relative_error",
            [
                {"id": "err1", "type": "absolute_error", "role": "given", "value": "<number>", "unit": "<same as measured or blank>"},
                {"id": "measured1", "type": "measured_value", "role": "given", "value": "<number>", "unit": "<unit or blank>"},
                {"id": "percent_query", "type": "percent_error", "role": "query", "value": None, "unit": "%"},
            ],
        ),
    ),
    FormulaDoc(
        id="harmonic_current_cos_time",
        name="Sinusoidal current at time t",
        equation="i(t) = I0*cos(omega*t + phi)",
        description="Use for instantaneous harmonic current from amplitude, angular frequency, time, and optional phase.",
        quantity_types=["current", "current_amplitude", "angular_frequency", "time", "phase"],
        query_types=["current"],
        tags=["sinusoidal", "time_domain", "current", "angular_frequency"],
        keywords=["instantaneous current", "cos omega t", "sinusoidal current", "current at time"],
        schema_template=_schema(
            "harmonic_current_cos_time",
            [
                {"id": "I0", "type": "current_amplitude", "role": "given", "value": "<number>", "unit": "<A|mA>"},
                {"id": "omega1", "type": "angular_frequency", "role": "given", "value": "<number>", "unit": "rad/s"},
                {"id": "t1", "type": "time", "role": "given", "value": "<number>", "unit": "<s|ms>"},
                {"id": "I_query", "type": "current", "role": "query", "value": None, "unit": "<A|mA>"},
            ],
        ),
    ),
]


FORMULA_BY_ID: dict[str, FormulaDoc] = {formula.id: formula for formula in FORMULA_DOCS}


def list_formula_docs() -> list[FormulaDoc]:
    return list(FORMULA_DOCS)


def get_formula_doc(formula_id: str) -> FormulaDoc:
    return FORMULA_BY_ID[formula_id]
