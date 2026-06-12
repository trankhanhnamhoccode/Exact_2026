from xai_physics.main import solve
from xai_physics.domains.capacitor_state.parallel_extractor import try_extract_parallel_problem


def test_extract_parallel_problem_two_capacitors():
    q = (
        "A 2 uF capacitor is charged to 100 V and disconnected. "
        "It is then connected in parallel to an uncharged 3 uF capacitor. "
        "Find the final voltage."
    )

    problem = try_extract_parallel_problem(q)

    assert problem is not None
    assert len(problem.system.capacitors) == 2
    assert problem.query == "voltage"

    c1 = problem.system.get("C1")
    c2 = problem.system.get("C2")

    assert abs(c1.capacitance_F - 2e-6) < 1e-18
    assert abs(c1.voltage_V - 100) < 1e-9
    assert abs(c2.capacitance_F - 3e-6) < 1e-18
    assert abs(c2.voltage_V - 0) < 1e-9


def test_end_to_end_parallel_redistribution_final_voltage():
    q = (
        "A 2 uF capacitor is charged to 100 V and disconnected. "
        "It is then connected in parallel to an uncharged 3 uF capacitor. "
        "Find the final voltage."
    )

    result = solve(q)

    assert result.status == "solved"
    assert result.domain == "capacitor_state"
    assert result.answer == "40 V"
