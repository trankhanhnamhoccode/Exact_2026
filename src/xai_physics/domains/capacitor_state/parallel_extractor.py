from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from xai_physics.core.units import to_si
from xai_physics.domains.capacitor_state.state import CapacitorState
from xai_physics.domains.capacitor_state.system import SystemState
from xai_physics.domains.capacitor_state.redistribution import ParallelRedistribution


NUM = r"[-+]?\d+(?:\.\d+)?"

CAP_UNITS = ["pF", "nF", "uF", "μF", "mF", "F"]
VOLT_UNITS = ["V", "mV", "kV"]


@dataclass
class ParallelProblem:
    system: SystemState
    events: list
    query: str


def _unit_pattern(units: list[str]) -> str:
    return "|".join(re.escape(u) for u in units)


def _find_capacitances_F(text: str) -> list[float]:
    unit_pat = _unit_pattern(CAP_UNITS)
    values: list[float] = []

    for m in re.finditer(rf"({NUM})\s*({unit_pat})", text, flags=re.I):
        value = float(m.group(1))
        unit = m.group(2)
        values.append(to_si(value, unit))

    return values


def _find_first_voltage_V(text: str) -> Optional[float]:
    unit_pat = _unit_pattern(VOLT_UNITS)

    # Examples:
    # "charged to 100 V"
    # "connected to a 100 V battery"
    # "100 V battery"
    patterns = [
        rf"charged\s+to\s+({NUM})\s*({unit_pat})",
        rf"connected\s+to\s+(?:a\s+)?({NUM})\s*({unit_pat})",
        rf"({NUM})\s*({unit_pat})\s*(?:battery|source)",
        rf"({NUM})\s*({unit_pat})",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return to_si(float(m.group(1)), m.group(2))

    return None


def _detect_query(text: str) -> str:
    lower = text.lower()

    if "charge" in lower:
        return "charge"
    if "energy" in lower:
        return "energy"
    if "capacitance" in lower:
        return "capacitance"

    return "voltage"


def try_extract_parallel_problem(question: str) -> Optional[ParallelProblem]:
    text = question.strip()
    lower = text.lower()

    if "parallel" not in lower:
        return None

    if "capacitor" not in lower:
        return None

    capacitances = _find_capacitances_F(text)
    if len(capacitances) < 2:
        return None

    initial_voltage = _find_first_voltage_V(text)
    if initial_voltage is None:
        return None

    # v0 scope:
    # support the common textbook case:
    # C1 is charged, disconnected, then connected in parallel to an uncharged C2.
    if "uncharged" not in lower:
        return None

    system = SystemState()

    c1 = CapacitorState(
        id="C1",
        capacitance_F=capacitances[0],
        voltage_V=initial_voltage,
        connected_to_source=False,
    )
    c1.infer_missing()

    c2 = CapacitorState(
        id="C2",
        capacitance_F=capacitances[1],
        voltage_V=0.0,
        connected_to_source=False,
    )
    c2.infer_missing()

    system.add(c1)
    system.add(c2)
    system.infer_all()

    return ParallelProblem(
        system=system,
        events=[ParallelRedistribution(capacitor_ids=["C1", "C2"])],
        query=_detect_query(text),
    )
