from __future__ import annotations


GEOMETRY_WORDS = [
    "point", "points", "line", "triangle", "equilateral",
    "collinear", "center", "midpoint", "distance between",
    "coordinate", "coordinates", "placed at"
]

ELECTROSTATIC_WORDS = [
    "charge", "charges", "electric force", "electrostatic force",
    "coulomb", "electric field"
]

STATE_WORDS = [
    "initially", "then", "after", "afterwards", "finally",
    "disconnected", "disconnect", "connected to source",
    "connected to battery", "battery", "insert dielectric",
    "dielectric is inserted", "plate separation", "area is doubled",
    "distance is doubled", "charge redistribution"
]


def route_domain(question: str) -> str:
    text = question.lower()

    has_geometry = any(word in text for word in GEOMETRY_WORDS)
    has_electrostatic = any(word in text for word in ELECTROSTATIC_WORDS)
    has_state = any(word in text for word in STATE_WORDS)

    if has_geometry and has_electrostatic:
        return "electrostatics"

    if has_state:
        return "capacitor_state"

    return "equations"
