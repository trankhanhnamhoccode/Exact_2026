from __future__ import annotations

import re

from xai_physics.domains.electrostatics.retrieval.types import TagHit


HARD_RULES: list[tuple[str, str, str]] = [
    ("resultant_vector", r"\b(resultant force|resultant vector|two electric forces|forces have magnitudes|same direction|opposite directions)\b", "Problem gives force vectors directly."),
    ("electric_field", r"\b(electric field|field strength|N/C|V/m)\b", "Problem asks for electric field."),
    ("net_force", r"\b(net force|force acting|electric force acting|force exerted|resultant force acting|force on|force acting on)\b", "Problem asks for net Coulomb force."),
    ("pairwise_distances", r"\b(AB|AC|BC|CA|CB|MA|MB)\s*=|\bseparated by\b.{0,80}\b(point C|AC|BC|CA|CB|perpendicular|midpoint|line segment)\b|\b\d+(?:\.\d+)?\s*(?:cm|mm|m)\s+apart\b", "Problem gives pairwise distances."),
    ("equilateral", r"\bequilateral\b|\bvertices of an equilateral triangle\b", "Equilateral triangle geometry."),
    ("isosceles_right", r"\bisosceles right\b|\bright[- ]?angle vertex\b|\bright[- ]?angled triangle\b", "Right/isosceles-right triangle geometry."),
    ("midpoint", r"\bmidpoint\b|\bmiddle point\b|\bplaced at H\b|\bmidpoint of (?:the )?(?:line segment )?(?:AB|connecting q1 and q2)\b", "Point is a midpoint."),
    ("perpendicular_bisector", r"\bperpendicular bisector\b|\bon the perpendicular bisector\b|\baway from AB\b|\bfrom the line segment AB\b", "Point lies on perpendicular bisector."),
    ("center", r"\bcenter\b|\bcentroid\b", "Problem mentions center of triangle."),
    ("collinear", r"\bcollinear\b|\bstraight line\b|\balong the line\b|\bon a line\b|\bplaced .* apart on a straight line\b|\bbetween\b", "Collinear geometry."),
    ("dielectric_medium", r"\b(dielectric constant|relative permittivity|epsilon|ε)\s*=", "Medium relative permittivity is given."),
    ("perpendicular_fields", r"\bfields?\b.{0,80}\bperpendicular\b|\bperpendicular\b.{0,80}\bfields?\b", "Field vectors are perpendicular."),
    ("foot_of_altitude", r"\bfoot of (the )?(altitude|perpendicular)\b", "Foot of perpendicular/altitude is given."),
    ("inverse_coulomb", r"\bfind q\b|q1\s*=\s*q2\s*=\s*q|exert(?:s)? a force", "Inverse Coulomb force asks for charge from force and distance."),
    ("inverse_angle", r"find the angle|angle between.*resultant|resultant.*also", "Inverse resultant-vector angle pattern."),
    ("opposite_sides", r"opposite sides", "Collinear sources on opposite sides of target."),
    ("point_on_line", r"along the line|line connecting|positioned along", "Point lies on a line segment with inferred distances."),
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

    tags = {h.tag for h in hits}

    if "perpendicular_bisector" in tags and "electric_field" in tags:
        add("field_on_perpendicular_bisector", "Electric field on perpendicular bisector pattern.", score=1.2)

    if "perpendicular_bisector" in tags and "net_force" in tags:
        add("perpendicular_bisector_force", "Net force on a charge placed on the perpendicular bisector.", score=1.25)

    if "equilateral" in tags and "net_force" in tags:
        add("equilateral_force", "Net force at a vertex or center of an equilateral triangle.", score=1.2)

    if "collinear" in tags and "net_force" in tags:
        add("collinear_force", "Net force for charges on one straight line.", score=1.2)

    if "inverse_coulomb" in tags:
        add("coulomb_equal_charge", "Find equal charges from Coulomb force and separation.", score=1.25)

    if "inverse_angle" in tags:
        add("resultant_angle", "Find angle from two forces and resultant magnitude.", score=1.25)

    if "opposite_sides" in tags and "net_force" in tags:
        add("target_centered_collinear_force", "Distances from target to sources on opposite sides are enough.", score=1.25)

    if "point_on_line" in tags and "net_force" in tags:
        add("point_on_segment_force", "Infer missing segment distance for point on line.", score=1.25)

    return hits
