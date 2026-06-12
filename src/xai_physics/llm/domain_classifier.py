from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_DOMAINS = {
    "capacitor_state",
    "electrostatics",
}


@dataclass
class DomainDecision:
    domain: str
    confidence: float
    tags: list[str]
    reason: str


CAPACITOR_KEYWORDS = {
    "capacitor": "capacitor",
    "capacitance": "capacitance",
    "dielectric": "dielectric",
    "battery": "source",
    "source": "source",
    "disconnected": "state_transition",
    "disconnect": "state_transition",
    "connected": "state_transition",
    "plate distance": "geometry_change",
    "plate separation": "geometry_change",
    "plate area": "geometry_change",
    "parallel": "connection",
    "uncharged": "charge_redistribution",
}


ELECTROSTATICS_KEYWORDS = {
    "charge": "charge",
    "charges": "charge",
    "electrostatic force": "force",
    "electric force": "force",
    "coulomb": "coulomb",
    "electric field": "field",
    "net force": "net_force",
    "point": "geometry",
    "points": "geometry",
    "vertices": "geometry",
    "triangle": "geometry",
    "equilateral": "equilateral_triangle",
    "collinear": "collinear",
    "line": "collinear",
    "distance": "distance",
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


def classify_domain(problem: str) -> DomainDecision:
    text = problem.lower()

    cap_score, cap_tags = _score_keywords(text, CAPACITOR_KEYWORDS)
    elec_score, elec_tags = _score_keywords(text, ELECTROSTATICS_KEYWORDS)

    # Bias correction:
    # "charge" appears in both domains, but "capacitor" should dominate capacitor_state.
    if "capacitor" in text:
        cap_score += 3

    # Electrostatics usually has spatial arrangement.
    if any(k in text for k in ["placed", "vertices", "triangle", "collinear", "at point"]):
        elec_score += 2

    if cap_score == 0 and elec_score == 0:
        return DomainDecision(
            domain="unknown",
            confidence=0.0,
            tags=[],
            reason="No supported domain keywords found.",
        )

    if cap_score >= elec_score:
        total = cap_score + elec_score
        confidence = cap_score / total if total else 0.0
        return DomainDecision(
            domain="capacitor_state",
            confidence=confidence,
            tags=cap_tags,
            reason=f"capacitor_state score={cap_score}, electrostatics score={elec_score}",
        )

    total = cap_score + elec_score
    confidence = elec_score / total if total else 0.0
    return DomainDecision(
        domain="electrostatics",
        confidence=confidence,
        tags=elec_tags,
        reason=f"electrostatics score={elec_score}, capacitor_state score={cap_score}",
    )
