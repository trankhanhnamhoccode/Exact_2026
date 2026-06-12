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
        ("measurement_error", 1.5, r"relative error|percentage error|absolute error"),
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

    if (
        "capacitor" in text
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

    return scores
