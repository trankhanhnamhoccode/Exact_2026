from __future__ import annotations

from dataclasses import dataclass, field

from xai_physics.domains.capacitor_state.state import CapacitorState


@dataclass
class SystemState:
    capacitors: dict[str, CapacitorState] = field(default_factory=dict)

    def add(self, capacitor: CapacitorState) -> None:
        self.capacitors[capacitor.id] = capacitor

    def get(self, capacitor_id: str) -> CapacitorState:
        if capacitor_id not in self.capacitors:
            raise KeyError(f"Unknown capacitor: {capacitor_id}")
        return self.capacitors[capacitor_id]

    def infer_all(self) -> None:
        for cap in self.capacitors.values():
            cap.infer_missing()

    def clone(self) -> "SystemState":
        return SystemState(
            capacitors={
                cid: cap.clone()
                for cid, cap in self.capacitors.items()
            }
        )

    def total_charge_C(self, capacitor_ids: list[str]) -> float:
        total = 0.0
        for cid in capacitor_ids:
            cap = self.get(cid)
            cap.infer_missing()
            if cap.charge_C is None:
                raise ValueError(f"Charge is unknown for capacitor {cid}.")
            total += cap.charge_C
        return total

    def total_capacitance_F(self, capacitor_ids: list[str]) -> float:
        total = 0.0
        for cid in capacitor_ids:
            cap = self.get(cid)
            if cap.capacitance_F is None:
                raise ValueError(f"Capacitance is unknown for capacitor {cid}.")
            total += cap.capacitance_F
        return total
