from xai_physics.domains.capacitor_state.state import CapacitorState
from xai_physics.domains.capacitor_state.events import (
    ConnectToSource,
    DisconnectFromSource,
    InsertDielectric,
    DistanceScale,
    AreaScale,
)
from xai_physics.main import solve


def test_disconnected_dielectric_voltage_halves_when_k_2():
    state = CapacitorState(
        id="C1",
        capacitance_F=500e-12,
        voltage_V=300,
        connected_to_source=True,
    )
    state.infer_missing()

    DisconnectFromSource().apply(state)
    InsertDielectric(dielectric_constant=2).apply(state)

    assert abs(state.voltage_V - 150) < 1e-9


def test_connected_dielectric_voltage_constant_when_k_2():
    state = CapacitorState(
        id="C1",
        capacitance_F=500e-12,
        voltage_V=300,
        connected_to_source=False,
    )

    ConnectToSource(voltage_V=300).apply(state)
    old_charge = state.charge_C

    InsertDielectric(dielectric_constant=2).apply(state)

    assert abs(state.voltage_V - 300) < 1e-9
    assert abs(state.capacitance_F - 1000e-12) < 1e-21
    assert abs(state.charge_C - 2 * old_charge) < 1e-18


def test_disconnected_distance_doubled_voltage_doubles():
    state = CapacitorState(
        id="C1",
        capacitance_F=500e-12,
        voltage_V=300,
        connected_to_source=True,
    )
    state.infer_missing()

    DisconnectFromSource().apply(state)
    DistanceScale(factor=2).apply(state)

    assert abs(state.capacitance_F - 250e-12) < 1e-21
    assert abs(state.voltage_V - 600) < 1e-9


def test_disconnected_area_doubled_voltage_halves():
    state = CapacitorState(
        id="C1",
        capacitance_F=500e-12,
        voltage_V=300,
        connected_to_source=True,
    )
    state.infer_missing()

    DisconnectFromSource().apply(state)
    AreaScale(factor=2).apply(state)

    assert abs(state.capacitance_F - 1000e-12) < 1e-21
    assert abs(state.voltage_V - 150) < 1e-9


def test_end_to_end_capacitor_state_voltage():
    q = (
        "A 500 pF capacitor is initially connected to a 300 V battery, "
        "then disconnected and a dielectric with constant 2 is inserted. "
        "Find the final voltage."
    )

    result = solve(q)

    assert result.status == "solved"
    assert result.domain == "capacitor_state"
    assert result.answer == "150 V"


def test_end_to_end_connected_dielectric_voltage_constant():
    q = (
        "A 500 pF capacitor is connected to a 300 V battery, "
        "then a dielectric with constant 2 is inserted while it remains connected. "
        "Find the final voltage."
    )

    result = solve(q)

    assert result.status == "solved"
    assert result.domain == "capacitor_state"
    assert result.answer == "300 V"


def test_end_to_end_disconnected_distance_doubled():
    q = (
        "A 500 pF capacitor is initially connected to a 300 V battery, "
        "then disconnected and the plate distance is doubled. "
        "Find the final voltage."
    )

    result = solve(q)

    assert result.status == "solved"
    assert result.domain == "capacitor_state"
    assert result.answer == "600 V"


def test_end_to_end_disconnected_area_doubled():
    q = (
        "A 500 pF capacitor is initially connected to a 300 V battery, "
        "then disconnected and the plate area is doubled. "
        "Find the final voltage."
    )

    result = solve(q)

    assert result.status == "solved"
    assert result.domain == "capacitor_state"
    assert result.answer == "150 V"
