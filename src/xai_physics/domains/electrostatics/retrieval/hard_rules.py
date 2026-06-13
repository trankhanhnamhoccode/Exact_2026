from __future__ import annotations

import re

from xai_physics.domains.electrostatics.retrieval.types import TagHit


HARD_RULES: list[tuple[str, str, str]] = [
    ("resultant_vector", r"\b(resultant force|resultant vector|two electric forces|forces have magnitudes|same direction|opposite directions)\b", "Problem gives force vectors directly."),
    ("electric_field", r"\b(electric field|field strength|N/C|V/m)\b", "Problem asks for electric field."),
    ("net_force", r"\b(net force|force acting|electric force acting|force exerted|resultant force acting)\b", "Problem asks for net Coulomb force."),
    ("pairwise_distances", r"\b(AB|AC|BC|CA|CB|MA|MB)\s*=|\bseparated by\b.{0,80}\b(point C|AC|BC|CA|CB)\b", "Problem gives pairwise distances."),
    ("equilateral", r"\bequilateral\b", "Equilateral triangle geometry."),
    ("isosceles_right", r"\bisosceles right\b|\bright[- ]?angle vertex\b|\bright[- ]?angled triangle\b", "Right/isosceles-right triangle geometry."),
    ("midpoint", r"\bmidpoint\b|\bmiddle point\b|\bplaced at H\b", "Point is a midpoint."),
    ("perpendicular_bisector", r"\bperpendicular bisector\b", "Point lies on perpendicular bisector."),
    ("center", r"\bcenter\b|\bcentroid\b", "Problem mentions center of triangle."),
    ("collinear", r"\bcollinear\b|\bstraight line\b|\balong the line\b|\bon a line\b|\bbetween\b", "Collinear geometry."),
    ("dielectric_medium", r"\b(dielectric constant|relative permittivity|epsilon|ε)\s*=", "Medium relative permittivity is given."),
    ("perpendicular_fields", r"\bfields?\b.{0,80}\bperpendicular\b|\bperpendicular\b.{0,80}\bfields?\b", "Field vectors are perpendicular."),
    ("foot_of_altitude", r"\bfoot of (the )?(altitude|perpendicular)\b", "Foot of perpendicular/altitude is given."),
]


def apply_hard_rules(problem: str) -> list[TagHit]:
    text = problem.lower()
    hits: list[TagHit] = []

    def add(tag: str, evidence: str, score: float = 1.0) -> None:
        if tag not in {hit.tag for hit in hits}:
            hits.append(TagHit(tag=tag, source="hard_rule", score=score, evidence=evidence))

    for tag, pattern, evidence in HARD_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            add(tag, evidence)

    if "midpoint" in {h.tag for h in hits} and "electric_field" in {h.tag for h in hits}:
        add("field_at_midpoint", "Electric field at midpoint pattern.", score=1.2)

    if "perpendicular_bisector" in {h.tag for h in hits} and "electric_field" in {h.tag for h in hits}:
        add("field_on_perpendicular_bisector", "Electric field on perpendicular bisector pattern.", score=1.2)

    return hits
