from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CapacitorState:
    id: str
    capacitance_F: Optional[float] = None
    voltage_V: Optional[float] = None
    charge_C: Optional[float] = None
    connected_to_source: bool = False

    def infer_missing(self) -> None:
        """
        Infer one missing value from Q = C * V if exactly two are known.
        """
        C = self.capacitance_F
        V = self.voltage_V
        Q = self.charge_C

        if C is not None and V is not None and Q is None:
            self.charge_C = C * V
        elif C is not None and Q is not None and V is None and C != 0:
            self.voltage_V = Q / C
        elif V is not None and Q is not None and C is None and V != 0:
            self.capacitance_F = Q / V

    def energy_J(self) -> Optional[float]:
        self.infer_missing()
        if self.capacitance_F is None or self.voltage_V is None:
            return None
        return 0.5 * self.capacitance_F * self.voltage_V ** 2

    def clone(self) -> "CapacitorState":
        return CapacitorState(
            id=self.id,
            capacitance_F=self.capacitance_F,
            voltage_V=self.voltage_V,
            charge_C=self.charge_C,
            connected_to_source=self.connected_to_source,
        )
