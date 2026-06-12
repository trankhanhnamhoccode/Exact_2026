from __future__ import annotations

from dataclasses import dataclass

from xai_physics.domains.capacitor_state.system import SystemState


@dataclass
class ParallelRedistribution:
    """
    Connect capacitors in parallel after they are isolated from external sources.

    Assumption:
    - Same polarity connection.
    - No external battery/source during redistribution.
    - Therefore total charge is conserved and final voltage is common.
    """

    capacitor_ids: list[str]
    name: str = "parallel_redistribution"

    def apply(self, system: SystemState) -> str:
        if len(self.capacitor_ids) < 2:
            raise ValueError("Parallel redistribution requires at least two capacitors.")

        system.infer_all()

        for cid in self.capacitor_ids:
            cap = system.get(cid)
            if cap.connected_to_source:
                raise ValueError(
                    f"Capacitor {cid} is still connected to a source. "
                    "This isolated redistribution event is not valid."
                )

        total_charge = system.total_charge_C(self.capacitor_ids)
        total_capacitance = system.total_capacitance_F(self.capacitor_ids)

        if total_capacitance == 0:
            raise ValueError("Total capacitance is zero.")

        final_voltage = total_charge / total_capacitance

        for cid in self.capacitor_ids:
            cap = system.get(cid)
            cap.voltage_V = final_voltage
            cap.charge_C = cap.capacitance_F * final_voltage
            cap.connected_to_source = False

        ids = ", ".join(self.capacitor_ids)
        return (
            f"Connected capacitors {ids} in parallel with same polarity. "
            "Since the system is isolated, total charge is conserved. "
            f"The final common voltage is V = Q_total / C_total = {final_voltage:g} V."
        )
