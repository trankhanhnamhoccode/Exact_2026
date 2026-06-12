from xai_physics.domains.capacitor_state.state import CapacitorState
from xai_physics.domains.capacitor_state.events import DisconnectFromSource, InsertDielectric
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
