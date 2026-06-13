from __future__ import annotations

import re

from .types import TagHit


def _add_tag(hits: list[TagHit], tag: str, score: float, evidence: str) -> None:
    if not any(hit.tag == tag for hit in hits):
        hits.append(TagHit(tag=tag, score=score, evidence=evidence))


def detect_tags(problem: str) -> list[TagHit]:
    text = problem.lower()
    hits: list[TagHit] = []

    patterns = [
        ("capacitor", 1.0, r"\bcapacitor|capacitance\b"),
        ("energy", 1.0, r"\benergy|stored energy\b"),
        ("voltage", 1.0, r"\bvoltage|potential difference|\bU\s*=|\bV\b"),
        ("charge", 1.0, r"\bcharge|\bQ\s*="),
        ("parallel_plate", 1.2, r"parallel[- ]plate|plate area|plate separation"),
        ("series", 1.2, r"\bseries\b"),
        ("lc", 1.2, r"\bLC\b|oscillat|tank circuit"),
        ("resonance", 1.5, r"resonan|natural frequency"),
        ("frequency", 1.0, r"\bfrequency|\bf\s*="),
        ("ohm", 1.3, r"\bohm|resistance|resistor|\bR\s*="),
        ("current", 1.0, r"\bcurrent|\bI\s*="),
        ("rlc", 1.2, r"\bRLC\b|quality factor|Q factor"),
        ("solenoid", 1.8, r"\bsolenoid\b"),
        ("magnetic_field", 1.2, r"magnetic field|\bB\s*="),
        ("point_charge", 1.4, r"point charge"),
        ("electric_field", 1.2, r"electric field|\bE\s*="),
        ("measurement_error", 1.5, r"relative error|percentage error|absolute error|least count|average absolute error"),
        ("least_count", 1.3, r"least count"),
        ("repeated_measurements", 1.2, r"average value|average absolute error|mean value|repeated measurements"),
        ("force", 1.2, r"attractive force|force between|force on"),
        ("time_domain", 1.2, r"instantaneous|at time|cos|sin|omega|angular frequency"),
    ]

    for tag, score, pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            _add_tag(hits, tag, score, pattern)

    return hits


def formula_rule_scores(problem: str) -> dict[str, float]:
    text = problem.lower()
    scores: dict[str, float] = {}

    def add(formula_id: str, score: float) -> None:
        scores[formula_id] = max(scores.get(formula_id, 0.0), score)

    is_scaling_question = (
        any(
            phrase in text
            for phrase in [
                "how many times",
                "by what factor",
                "doubled",
                "tripled",
                "halved",
                "constant voltage",
                "constant capacitance",
                "change",
                "changes",
            ]
        )
        or re.search(r"\bratio\b", text) is not None
    )

    if (
        not is_scaling_question
        and "capacitor" in text
        and ("energy" in text or "stored" in text)
        and ("capacitance" in text or re.search(r"\bc\s*=", text))
        and ("voltage" in text or "potential difference" in text or re.search(r"\bu\s*=", text))
    ):
        add("capacitor_energy_voltage", 5.0)

    if "capacitor" in text and "charge" in text and ("voltage" in text or "potential difference" in text):
        add("capacitor_charge_voltage", 5.0)

    if ("parallel plate" in text or "parallel-plate" in text or "plate area" in text or "plate separation" in text) and "capacitance" in text:
        add("parallel_plate_capacitance", 5.0)

    if "series" in text and "capacitor" in text and ("equivalent" in text or "unknown" in text):
        add("series_capacitance_unknown", 5.0)

    if ("resonant" in text or "resonance" in text or "natural frequency" in text) and ("lc" in text or "inductance" in text or "capacitor" in text):
        add("lc_resonance_frequency", 5.0)

    if ("quality factor" in text or "q factor" in text) and ("rlc" in text or "inductance" in text):
        add("quality_factor", 5.0)

    if ("ohm" in text or "resistance" in text or "resistor" in text) and ("current" in text or "voltage" in text):
        add("ohm_law", 4.5)

    if "solenoid" in text and "magnetic field" in text:
        add("solenoid_magnetic_field", 5.0)

    if "electric field" in text and "point charge" in text:
        add("point_charge_electric_field", 5.0)

    if "relative error" in text or "percentage error" in text:
        add("percentage_relative_error", 5.0)

    if ("instantaneous current" in text or "current at time" in text) and ("cos" in text or "omega" in text or "angular frequency" in text):
        add("harmonic_current_cos_time", 5.0)

    if "electric field" in text and ("between plates" in text or "plate separation" in text) and ("voltage" in text or re.search(r"\bu\s*=", text)):
        add("parallel_plate_field", 5.0)

    if "energy density" in text and ("capacitor" in text or "electric field" in text or "between plates" in text):
        add("capacitor_energy_density", 5.0)

    # Important wording trap: many textbook translations say "electric field energy"
    # for the stored energy of a capacitor, not energy density.
    if "electric field energy" in text and "capacitor" in text and "energy density" not in text:
        add("capacitor_energy_voltage", 6.0)

    if "constant voltage" in text and "energy" in text and ("capacitance" in text or "capacitor" in text):
        add("capacitor_energy_scaling_constant_voltage", 5.0)

    if "constant capacitance" in text and "energy" in text and "voltage" in text:
        add("capacitor_energy_voltage_scaling_constant_capacitance", 5.0)

    # Voltage-ratio scaling trap:
    # "electric field energy" often triggers ordinary capacitor energy, but if the
    # question asks how many times/by what factor after voltage is doubled/tripled/etc.,
    # the correct schema should use U_ratio and W_ratio_query.
    has_voltage_ratio_change = (
        re.search(r"(?:voltage|potential difference|\bu\b).{0,40}(?:doubled|tripled|halved|increases|decreases|increased|decreased|multiplied)", text)
        is not None
        or re.search(r"(?:doubled|tripled|halved).{0,40}(?:voltage|potential difference|\bu\b)", text)
        is not None
    )
    if (
        is_scaling_question
        and has_voltage_ratio_change
        and ("constant voltage" not in text)
        and ("capacitor" in text or "capacitance" in text)
        and ("energy" in text or "stored" in text or "electric field energy" in text)
        and "energy density" not in text
    ):
        add("capacitor_energy_voltage_scaling_constant_capacitance", 8.0)

    if "inductor" in text and ("energy" in text or "magnetic energy" in text):
        add("inductor_energy", 5.0)

    if "impedance" in text and ("reactance" in text or "xl" in text or "xc" in text):
        add("ac_impedance", 5.0)

    if "power" in text and "voltage" in text and ("resistance" in text or "resistor" in text):
        add("power_voltage_resistance", 5.0)

    if "magnetic flux" in text or "flux linkage" in text:
        add("magnetic_flux_total", 5.0)

    if "absolute error" in text and ("actual" in text or "measured" in text):
        add("absolute_error_from_actual", 5.0)

    if "lc" in text and "energy" in text and ("time" in text or "omega" in text or "cos" in text):
        add("lc_electric_energy_time", 4.5)

    if "capacitor" in text and "energy" in text and "charge" in text and ("voltage" in text or "potential difference" in text):
        add("capacitor_energy_charge_voltage", 5.5)

    if ("distance" in text or "separation" in text) and ("halved" in text or "doubled" in text) and "capacitance" in text:
        add("parallel_plate_capacitance_distance_scaling", 5.5)

    if ("attractive force" in text or "force between" in text) and "plate" in text and ("charge" in text or "q" in text) and ("area" in text or "s =" in text):
        add("capacitor_plate_force_by_charge_area", 5.5)

    if "least count" in text and "absolute error" in text:
        add("instrument_absolute_error", 5.5)

    if "average absolute error" in text or "average value" in text or "mean value" in text:
        add("measurement_average", 5.5)

    if ("is" in text and "resonance" in text) or "in resonance at" in text:
        add("resonance_check", 5.5)

    if "parallel" in text and ("resistance" in text or "resistor" in text or "branches" in text):
        add("parallel_resistance", 5.5)

    return scores
