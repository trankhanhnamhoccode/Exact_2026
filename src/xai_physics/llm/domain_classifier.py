from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_DOMAINS = {
    "capacitor_state",
    "electrostatics",
    "equations",
}


@dataclass
class DomainDecision:
    domain: str
    confidence: float
    tags: list[str]
    reason: str


CAPACITOR_BASE_KEYWORDS = {
    "capacitor": "capacitor",
    "capacitance": "capacitance",
}


CAPACITOR_STATE_KEYWORDS = {
    "initially": "state_transition",
    "then": "state_transition",
    "after": "state_transition",
    "afterwards": "state_transition",
    "finally": "state_transition",
    "disconnected": "disconnect",
    "disconnect": "disconnect",
    "removed from the battery": "disconnect",
    "battery is removed": "disconnect",
    "isolated": "disconnect",
    "connected to source": "source",
    "connected to battery": "source",
    "insert dielectric": "dielectric_event",
    "dielectric is inserted": "dielectric_event",
    "inserted between": "dielectric_event",
    "dielectric is removed": "dielectric_event",
    "replace dielectric": "replace_dielectric",
    "replaced by dielectric": "replace_dielectric",
    "replaced by another dielectric": "replace_dielectric",
    "replaced": "replace_dielectric",
    "plate separation is doubled": "distance_scale",
    "distance is doubled": "distance_scale",
    "separation is doubled": "distance_scale",
    "area is doubled": "area_scale",
    "short-circuited": "short_circuit",
    "short circuit": "short_circuit",
    "connected to an inductor": "lc_event",
    "uncharged capacitor": "charge_redistribution",
    "connected in parallel": "charge_redistribution",
    "charge redistribution": "charge_redistribution",
}


ELECTROSTATICS_KEYWORDS = {
    "electrostatic force": "force",
    "electric force": "force",
    "coulomb": "coulomb",
    "net force": "net_force",
    "force acting": "net_force",
    "point charge": "point_charge",
    "charges are placed": "geometry",
    "placed at points": "geometry",
    "vertices": "geometry",
    "triangle": "geometry",
    "equilateral": "equilateral_triangle",
    "collinear": "collinear",
    "straight line": "collinear",
    "coordinate": "coordinate",
    "coordinates": "coordinate",
}


EQUATIONS_KEYWORDS = {
    "capacitor": "capacitor",
    "capacitance": "capacitance",
    "voltage": "voltage",
    "charge": "charge",
    "energy": "energy",
    "inductor": "inductor",
    "inductance": "inductance",
    "current": "current",
    "resistance": "resistance",
    "impedance": "impedance",
    "rlc": "rlc",
    "resonance": "resonance",
    "resonant": "resonance",
    "frequency": "frequency",
    "solenoid": "solenoid",
    "magnetic field": "magnetic_field",
    "magnetic flux": "magnetic_flux",
    "quality factor": "quality_factor",
    "power": "power",
    "ohm": "ohm",
    "electric field": "electric_field",
    "relative error": "measurement_error",
    "absolute error": "measurement_error",
}


def _score_keywords(text: str, keyword_map: dict[str, str]) -> tuple[int, list[str]]:
    score = 0
    tags: list[str] = []

    for keyword, tag in keyword_map.items():
        if keyword in text:
            score += 1
            if tag not in tags:
                tags.append(tag)

    return score, tags


def _add_unique(tags: list[str], tag: str) -> None:
    if tag not in tags:
        tags.append(tag)


def classify_domain(problem: str) -> DomainDecision:
    text = problem.lower()

    cap_base_score, cap_base_tags = _score_keywords(text, CAPACITOR_BASE_KEYWORDS)
    state_score, state_tags = _score_keywords(text, CAPACITOR_STATE_KEYWORDS)
    elec_score, elec_tags = _score_keywords(text, ELECTROSTATICS_KEYWORDS)
    eq_score, eq_tags = _score_keywords(text, EQUATIONS_KEYWORDS)

    has_capacitor = "capacitor" in text or "capacitance" in text
    has_dielectric_replace = (
        has_capacitor
        and "dielectric" in text
        and ("replace" in text or "replaced" in text)
    )

    if has_capacitor:
        _add_unique(state_tags, "capacitor")

    # Strong electrostatics boost only when charges have spatial arrangement.
    if any(k in text for k in ["placed", "vertices", "triangle", "collinear", "at point", "coordinates"]):
        if any(k in text for k in ["charge", "charges", "electric force", "electrostatic force", "net force"]):
            elec_score += 3

    # Capacitor state requires a real event/state transition.
    # Plain capacitor formulas stay equations.
    if has_capacitor and (state_score > 0 or has_dielectric_replace):
        state_score += 3 + cap_base_score
        if has_dielectric_replace:
            _add_unique(state_tags, "replace_dielectric")
        return DomainDecision(
            domain="capacitor_state",
            confidence=state_score / max(state_score + elec_score + eq_score, 1),
            tags=state_tags,
            reason=(
                f"capacitor_state score={state_score}, "
                f"electrostatics score={elec_score}, "
                f"equations score={eq_score}"
            ),
        )

    # Plain scalar formulas should prefer equations.
    if eq_score > 0 and elec_score == 0:
        return DomainDecision(
            domain="equations",
            confidence=1.0,
            tags=eq_tags,
            reason=f"equations score={eq_score}, no state/electrostatic geometry trigger",
        )

    scores = {
        "electrostatics": elec_score,
        "equations": eq_score,
    }

    best_domain = max(scores, key=scores.get)
    best_score = scores[best_domain]

    if best_score == 0:
        return DomainDecision(
            domain="unknown",
            confidence=0.0,
            tags=[],
            reason="No supported domain keywords found.",
        )

    total = sum(scores.values())
    confidence = best_score / total if total else 0.0

    tags_by_domain = {
        "electrostatics": elec_tags,
        "equations": eq_tags,
    }

    return DomainDecision(
        domain=best_domain,
        confidence=confidence,
        tags=tags_by_domain[best_domain],
        reason=(
            f"capacitor_state score={state_score}, "
            f"electrostatics score={elec_score}, "
            f"equations score={eq_score}"
        ),
    )
