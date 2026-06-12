from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from xai_physics.core.units import to_si
from xai_physics.domains.capacitor_state.state import CapacitorState
from xai_physics.domains.capacitor_state.events import (
    AreaScale,
    ConnectToSource,
    DisconnectFromSource,
    DistanceScale,
    InsertDielectric,
)


NUM = r"[-+]?\d+(?:\.\d+)?"


@dataclass
class CapacitorProblem:
    initial_state: CapacitorState
    events: list
    query: str


def _find_quantity(text: str, units: list[str]) -> Optional[tuple[float, str]]:
    unit_pat = "|".join(re.escape(u) for u in units)
    m = re.search(rf"({NUM})\s*({unit_pat})", text, flags=re.I)
    if not m:
        return None
    return float(m.group(1)), m.group(2)


def _find_capacitance_F(text: str) -> Optional[float]:
    found = _find_quantity(text, ["pF", "nF", "uF", "μF", "mF", "F"])
    if not found:
        return None
    value, unit = found
    return to_si(value, unit)


def _find_voltage_V(text: str) -> Optional[float]:
    # Prefer voltage near battery/source.
    m = re.search(rf"({NUM})\s*(V|mV|kV)\s*(?:battery|source)", text, flags=re.I)
    if not m:
        m = re.search(rf"(?:battery|source).*?({NUM})\s*(V|mV|kV)", text, flags=re.I)
    if not m:
        found = _find_quantity(text, ["V", "mV", "kV"])
        if not found:
            return None
        value, unit = found
        return to_si(value, unit)

    return to_si(float(m.group(1)), m.group(2))


def _find_dielectric_constant(text: str) -> Optional[float]:
    patterns = [
        rf"dielectric(?:\s+constant)?\s*(?:k|κ)?\s*(?:=|of|with constant)?\s*({NUM})",
        rf"k\s*=\s*({NUM})",
        rf"constant\s*({NUM})",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return float(m.group(1))

    return None


def _find_scale_factor(text: str, keyword: str) -> Optional[float]:
    # Examples:
    # "plate distance is doubled"
    # "plate separation is tripled"
    # "area is doubled"
    lower = text.lower()

    if keyword in lower:
        if "doubled" in lower:
            return 2.0
        if "tripled" in lower:
            return 3.0
        if "halved" in lower:
            return 0.5

    m = re.search(rf"{keyword}.*?(?:scaled|multiplied|increased)\s+by\s+({NUM})", text, flags=re.I)
    if m:
        return float(m.group(1))

    return None


def _detect_query(text: str) -> str:
    lower = text.lower()

    if "energy" in lower:
        return "energy"
    if "charge" in lower:
        return "charge"
    if "capacitance" in lower:
        return "capacitance"
    if "voltage" in lower or "potential difference" in lower:
        return "voltage"

    return "voltage"


def extract_problem(question: str) -> CapacitorProblem:
    text = question.strip()

    capacitance_F = _find_capacitance_F(text)
    voltage_V = _find_voltage_V(text)

    # If no capacitance is provided, use 1 F as symbolic scale for voltage-ratio problems.
    # This is okay for questions like final voltage after disconnected + dielectric insertion.
    if capacitance_F is None:
        capacitance_F = 1.0

    state = CapacitorState(
        id="C1",
        capacitance_F=capacitance_F,
        voltage_V=voltage_V,
        connected_to_source=False,
    )
    state.infer_missing()

    events = []

    lower = text.lower()

    if "connected" in lower and ("battery" in lower or "source" in lower):
        if voltage_V is not None:
            events.append(ConnectToSource(voltage_V=voltage_V))

    if "disconnect" in lower or "disconnected" in lower:
        events.append(DisconnectFromSource())

    k = _find_dielectric_constant(text)
    if k is not None and ("dielectric" in lower or "k =" in lower):
        events.append(InsertDielectric(dielectric_constant=k))

    d_factor = _find_scale_factor(text, "distance")
    if d_factor is None:
        d_factor = _find_scale_factor(text, "separation")
    if d_factor is not None:
        events.append(DistanceScale(factor=d_factor))

    a_factor = _find_scale_factor(text, "area")
    if a_factor is not None:
        events.append(AreaScale(factor=a_factor))

    query = _detect_query(text)

    return CapacitorProblem(
        initial_state=state,
        events=events,
        query=query,
    )
