from __future__ import annotations

import re

from xai_physics.domains.capacitor_state.retrieval.types import TagHit


HARD_RULES: list[tuple[str, str, str]] = [
    (
        "dielectric",
        r"\b(dielectric|dielectric constant|relative permittivity|permittivity|ε|epsilon|eps|k\s*=)\b",
        "Dielectric insertion or permittivity change is explicitly mentioned.",
    ),
    (
        "disconnect",
        r"\b(disconnected|disconnect|isolated|cut from the source|removed from (the )?(battery|source|power source))\b",
        "Capacitor is disconnected or isolated.",
    ),
    (
        "connected_source",
        r"\b(connected to (a )?(battery|source|power source)|remains connected|still connected|while still connected)\b",
        "Capacitor is connected to a source.",
    ),
    (
        "parallel_connection",
        r"\b(parallel|connected in parallel)\b|connected\s+(with|to)?\s*(another|their)?\s*.{0,80}\b(terminals|plates)\b.{0,80}\btogether\b",
        "Parallel-like capacitor connection is explicitly mentioned.",
    ),
    (
        "like_polarity_connection",
        r"\b(like[- ]?(poled|polarity|signed)|same[- ]?(polarity|signed)|positive to positive|negative to negative)\b",
        "Like-polarity terminal connection is explicitly mentioned.",
    ),
    (
        "uncharged_capacitor",
        r"\b(uncharged|initially uncharged)\b",
        "Uncharged capacitor is explicitly mentioned.",
    ),
    (
        "distance_scale",
        r"\b(plate distance|plate separation|distance between plates|separation|plates are moved apart|moved apart|distance).{0,80}\b(doubled|tripled|quadrupled|halved|increased|decreased|scaled|4 times)\b",
        "Plate distance/separation changes.",
    ),
    (
        "area_scale",
        r"\b(plate area|area of plates|area).{0,80}\b(doubled|tripled|quadrupled|halved|increased|decreased|scaled)\b",
        "Plate area changes.",
    ),
    (
        "capacitance_scale",
        r"\b(capacitance|capacity).{0,80}\b(doubled|tripled|quadrupled|halved|increased|decreased|scaled)\b",
        "Capacitance changes by a stated factor.",
    ),
    (
        "short_circuit",
        r"\b(short[- ]?circuited|short circuit)\b",
        "Capacitor is short-circuited.",
    ),
    (
        "series_connection",
        r"\b(series|connected in series)\b",
        "Series connection is mentioned.",
    ),
    (
        "inductor_oscillation",
        r"\b(inductor|oscillation|LC|oscillating circuit)\b",
        "Capacitor-inductor oscillation is mentioned.",
    ),
    (
        "work_query",
        r"\b(work|additional work|work supplied)\b",
        "The problem asks about work.",
    ),
    (
        "ratio_query",
        r"\b(how many times|percentage|percent|how does .* change|increase|decrease)\b",
        "The problem asks for a ratio or qualitative change.",
    ),
    (
        "charge_query",
        r"\b(find|calculate|determine|what is).{0,80}\b(charge)\b",
        "The problem asks for charge.",
    ),
    (
        "voltage_query",
        r"\b(find|calculate|determine|what is).{0,100}\b(voltage|potential difference)\b",
        "The problem asks for voltage.",
    ),
    (
        "energy_query",
        r"\b(find|calculate|determine|what is).{0,100}\b(energy|electric field energy|stored energy)\b",
        "The problem asks for energy.",
    ),
    (
        "capacitance_query",
        r"\b(find|calculate|determine|what is).{0,100}\b(capacitance)\b",
        "The problem asks for capacitance.",
    ),
]


DERIVED_RULES: list[tuple[str, set[str], str]] = [
    (
        "charge_redistribution",
        {"parallel_connection", "uncharged_capacitor"},
        "Parallel connection with an uncharged capacitor indicates charge redistribution.",
    ),
    (
        "charge_redistribution",
        {"parallel_connection", "like_polarity_connection"},
        "Like-polarity capacitor connection indicates charge redistribution.",
    ),
    (
        "isolated_dielectric",
        {"disconnect", "dielectric"},
        "Disconnected capacitor plus dielectric insertion implies charge conservation.",
    ),
    (
        "source_dielectric",
        {"connected_source", "dielectric"},
        "Connected source plus dielectric insertion implies voltage stays fixed.",
    ),
    (
        "isolated_distance_change",
        {"disconnect", "distance_scale"},
        "Disconnected capacitor plus plate distance change implies charge conservation.",
    ),
    (
        "source_distance_change",
        {"connected_source", "distance_scale"},
        "Connected source plus plate distance change implies voltage stays fixed.",
    ),
    (
        "source_capacitance_change",
        {"connected_source", "capacitance_scale"},
        "Connected source plus capacitance change implies voltage stays fixed.",
    ),
]


def apply_hard_rules(problem: str) -> list[TagHit]:
    text = problem.lower()
    hits: list[TagHit] = []

    def add(
        tag: str,
        score: float,
        evidence: str,
        source: str = "hard_rule",
    ) -> None:
        if any(hit.tag == tag for hit in hits):
            return

        hits.append(
            TagHit(
                tag=tag,
                source=source,
                score=score,
                evidence=evidence,
            )
        )

    for tag, pattern, evidence in HARD_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            add(
                tag=tag,
                source="hard_rule",
                score=1.0,
                evidence=evidence,
            )

    existing = {hit.tag for hit in hits}

    for tag, required, evidence in DERIVED_RULES:
        if required.issubset(existing) and tag not in existing:
            add(
                tag=tag,
                source="derived_rule",
                score=1.0,
                evidence=evidence,
            )
            existing.add(tag)


    if (
        ("voltage" in text or "potential difference" in text)
        and ("kept constant" in text or "kept fixed" in text or "constant voltage" in text)
    ):
        add(
            tag="connected_source",
            source="hard_rule",
            score=1.05,
            evidence="constant-voltage wording implies source-held voltage",
        )

    if (
        ("work" in text or "supplied by the source" in text)
        and ("source" in text or "battery" in text)
    ):
        add(
            tag="source_work_query",
            source="hard_rule",
            score=1.10,
            evidence="work supplied by source query",
        )

    if (
        ("replace" in text or "replaced" in text)
        and ("dielectric" in text or "permittivity" in text)
    ):
        add(
            tag="replace_dielectric",
            source="hard_rule",
            score=0.95,
            evidence="dielectric replacement language",
        )

    if (
        ("ratio" in text or "how many times" in text or "times" in text)
        and ("capacitance" in text or "capacity" in text)
    ):
        add(
            tag="capacitance_ratio_query",
            source="hard_rule",
            score=0.95,
            evidence="capacitance ratio query",
        )
        add(
            tag="ratio_query",
            source="hard_rule",
            score=0.90,
            evidence="ratio query",
        )


    if (
        ("percent" in text or "percentage" in text or "%" in text)
        and ("energy" in text or "stored energy" in text or "electric field energy" in text)
    ):
        add(
            tag="energy_percent_query",
            source="hard_rule",
            score=0.95,
            evidence="energy percentage query",
        )
        add(
            tag="ratio_query",
            source="hard_rule",
            score=0.90,
            evidence="percentage is a ratio query",
        )


    if (
        ("replace" in text or "replaced" in text)
        and "capacitor" in text
        and "another" in text
        and "dielectric" not in text
        and "permittivity" not in text
    ):
        add(
            tag="replace_capacitor",
            source="hard_rule",
            score=1.20,
            evidence="capacitor replacement language, not dielectric replacement",
        )
        add(
            tag="capacitance_change",
            source="hard_rule",
            score=1.05,
            evidence="new capacitor capacitance changes the state",
        )

    if (
        ("reduction in energy" in text or "energy reduction" in text or "decrease in energy" in text)
        and "capacitor" in text
    ):
        add(
            tag="energy_reduction_query",
            source="hard_rule",
            score=1.10,
            evidence="question asks for absolute energy reduction",
        )

    if (
        ("same voltage" in text or "maintaining the same voltage" in text or "voltage is kept" in text)
        and "capacitor" in text
    ):
        add(
            tag="constant_voltage",
            source="hard_rule",
            score=1.05,
            evidence="replacement/change keeps voltage constant",
        )

    return hits
