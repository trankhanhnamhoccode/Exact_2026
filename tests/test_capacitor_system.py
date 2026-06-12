from xai_physics.domains.capacitor_state.state import CapacitorState
from xai_physics.domains.capacitor_state.system import SystemState
from xai_physics.domains.capacitor_state.redistribution import ParallelRedistribution


def test_system_state_total_charge_and_capacitance():
    system = SystemState()
    system.add(CapacitorState(id="C1", capacitance_F=2e-6, voltage_V=100))
    system.add(CapacitorState(id="C2", capacitance_F=3e-6, voltage_V=0))

    system.infer_all()

    assert abs(system.total_charge_C(["C1", "C2"]) - 2e-4) < 1e-12
    assert abs(system.total_capacitance_F(["C1", "C2"]) - 5e-6) < 1e-18


def test_parallel_redistribution_two_capacitors():
    system = SystemState()
    system.add(
        CapacitorState(
            id="C1",
            capacitance_F=2e-6,
            voltage_V=100,
            connected_to_source=False,
        )
    )
    system.add(
        CapacitorState(
            id="C2",
            capacitance_F=3e-6,
            voltage_V=0,
            connected_to_source=False,
        )
    )

    system.infer_all()

    event = ParallelRedistribution(capacitor_ids=["C1", "C2"])
    detail = event.apply(system)

    c1 = system.get("C1")
    c2 = system.get("C2")

    assert "final common voltage" in detail
    assert abs(c1.voltage_V - 40) < 1e-9
    assert abs(c2.voltage_V - 40) < 1e-9
    assert abs(c1.charge_C - 80e-6) < 1e-12
    assert abs(c2.charge_C - 120e-6) < 1e-12


def test_parallel_redistribution_rejects_connected_source():
    system = SystemState()
    system.add(
        CapacitorState(
            id="C1",
            capacitance_F=2e-6,
            voltage_V=100,
            connected_to_source=True,
        )
    )
    system.add(
        CapacitorState(
            id="C2",
            capacitance_F=3e-6,
            voltage_V=0,
            connected_to_source=False,
        )
    )

    event = ParallelRedistribution(capacitor_ids=["C1", "C2"])

    try:
        event.apply(system)
    except ValueError as exc:
        assert "still connected to a source" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
