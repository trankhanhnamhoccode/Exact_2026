from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from xai_physics.domains.capacitor_state.state import CapacitorState


class Event(Protocol):
    name: str

    def apply(self, state: CapacitorState) -> str:
        ...


@dataclass
class ConnectToSource:
    voltage_V: float
    name: str = "connect_to_source"

    def apply(self, state: CapacitorState) -> str:
        state.voltage_V = self.voltage_V
        state.connected_to_source = True
        state.charge_C = None
        state.infer_missing()
        return (
            f"Connected capacitor {state.id} to a voltage source of "
            f"{self.voltage_V:g} V. Voltage is fixed by the source."
        )


@dataclass
class DisconnectFromSource:
    name: str = "disconnect_from_source"

    def apply(self, state: CapacitorState) -> str:
        state.infer_missing()
        state.connected_to_source = False
        return (
            f"Disconnected capacitor {state.id} from the source. "
            "After disconnection, charge is conserved unless another connection changes it."
        )


@dataclass
class InsertDielectric:
    dielectric_constant: float
    name: str = "insert_dielectric"

    def apply(self, state: CapacitorState) -> str:
        state.infer_missing()

        old_C = state.capacitance_F
        old_V = state.voltage_V
        old_Q = state.charge_C

        if old_C is None:
            raise ValueError("Cannot insert dielectric because capacitance is unknown.")

        new_C = old_C * self.dielectric_constant
        state.capacitance_F = new_C

        if state.connected_to_source:
            # Battery keeps voltage constant; charge changes.
            state.voltage_V = old_V
            state.charge_C = None
            state.infer_missing()
            return (
                f"Inserted dielectric k={self.dielectric_constant:g}. "
                "Because the capacitor is still connected to a source, voltage stays constant "
                "and charge changes according to Q = C V."
            )

        # Disconnected capacitor: charge is conserved.
        state.charge_C = old_Q
        state.voltage_V = None
        state.infer_missing()

        return (
            f"Inserted dielectric k={self.dielectric_constant:g}. "
            "Because the capacitor is disconnected, charge is conserved; "
            "capacitance increases and voltage changes by V = Q / C."
        )


@dataclass
class DistanceScale:
    factor: float
    name: str = "distance_scale"

    def apply(self, state: CapacitorState) -> str:
        state.infer_missing()

        if state.capacitance_F is None:
            raise ValueError("Cannot scale plate distance because capacitance is unknown.")

        old_Q = state.charge_C
        state.capacitance_F = state.capacitance_F / self.factor

        if state.connected_to_source:
            state.charge_C = None
            state.infer_missing()
            return (
                f"Plate distance scaled by {self.factor:g}. "
                "For a parallel-plate capacitor, C is inversely proportional to d. "
                "Since it is connected to a source, voltage stays fixed."
            )

        state.charge_C = old_Q
        state.voltage_V = None
        state.infer_missing()
        return (
            f"Plate distance scaled by {self.factor:g}. "
            "Since the capacitor is disconnected, charge is conserved."
        )


@dataclass
class AreaScale:
    factor: float
    name: str = "area_scale"

    def apply(self, state: CapacitorState) -> str:
        state.infer_missing()

        if state.capacitance_F is None:
            raise ValueError("Cannot scale plate area because capacitance is unknown.")

        old_Q = state.charge_C
        state.capacitance_F = state.capacitance_F * self.factor

        if state.connected_to_source:
            state.charge_C = None
            state.infer_missing()
            return (
                f"Plate area scaled by {self.factor:g}. "
                "For a parallel-plate capacitor, C is proportional to A. "
                "Since it is connected to a source, voltage stays fixed."
            )

        state.charge_C = old_Q
        state.voltage_V = None
        state.infer_missing()
        return (
            f"Plate area scaled by {self.factor:g}. "
            "Since the capacitor is disconnected, charge is conserved."
        )


class ShortCircuit:
    """
    Short-circuit a capacitor.

    Physical meaning:
    - The two plates are directly connected by a conducting path.
    - Final voltage becomes zero.
    - Final stored charge becomes zero.
    - Final electric field energy becomes zero.
    - Capacitance itself is unchanged.
    """

    def apply(self, cap):
        cap.voltage_V = 0.0
        cap.charge_C = 0.0
        cap.connected_to_source = False
        return cap
