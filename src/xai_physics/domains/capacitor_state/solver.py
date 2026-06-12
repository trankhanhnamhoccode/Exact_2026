from __future__ import annotations

from xai_physics.core.result import SolveResult
from xai_physics.domains.capacitor_state.extractor import extract_problem
from xai_physics.domains.capacitor_state.parallel_extractor import try_extract_parallel_problem


def _format_single_answer(query: str, state) -> str:
    state.infer_missing()

    if query == "voltage":
        if state.voltage_V is None:
            raise ValueError("Final voltage is unknown.")
        return f"{state.voltage_V:g} V"

    if query == "charge":
        if state.charge_C is None:
            raise ValueError("Final charge is unknown.")
        return f"{state.charge_C:g} C"

    if query == "capacitance":
        if state.capacitance_F is None:
            raise ValueError("Final capacitance is unknown.")
        return f"{state.capacitance_F:g} F"

    if query == "energy":
        energy = state.energy_J()
        if energy is None:
            raise ValueError("Final energy is unknown.")
        return f"{energy:g} J"

    raise ValueError(f"Unsupported query: {query}")


def _format_system_answer(query: str, system) -> str:
    system.infer_all()

    caps = list(system.capacitors.values())
    if not caps:
        raise ValueError("System has no capacitors.")

    # For same-polarity parallel redistribution, all capacitors share final voltage.
    if query == "voltage":
        voltage = caps[0].voltage_V
        if voltage is None:
            raise ValueError("Final voltage is unknown.")
        return f"{voltage:g} V"

    if query == "charge":
        total_charge = sum((cap.charge_C or 0.0) for cap in caps)
        return f"{total_charge:g} C"

    if query == "capacitance":
        total_capacitance = sum((cap.capacitance_F or 0.0) for cap in caps)
        return f"{total_capacitance:g} F"

    if query == "energy":
        total_energy = 0.0
        for cap in caps:
            e = cap.energy_J()
            if e is None:
                raise ValueError(f"Energy is unknown for capacitor {cap.id}.")
            total_energy += e
        return f"{total_energy:g} J"

    raise ValueError(f"Unsupported system query: {query}")


def _solve_parallel(question: str) -> SolveResult | None:
    problem = try_extract_parallel_problem(question)
    if problem is None:
        return None

    result = SolveResult(status="solved", domain="capacitor_state")
    system = problem.system.clone()
    system.infer_all()

    result.add_step(
        "Parallel redistribution problem detected",
        "Parsed a multi-capacitor isolated parallel redistribution problem.",
        capacitors={
            cid: {
                "capacitance_F": cap.capacitance_F,
                "voltage_V": cap.voltage_V,
                "charge_C": cap.charge_C,
                "connected_to_source": cap.connected_to_source,
            }
            for cid, cap in system.capacitors.items()
        },
    )

    for event in problem.events:
        detail = event.apply(system)
        system.infer_all()
        result.add_step(
            f"Apply event: {event.name}",
            detail,
            capacitors={
                cid: {
                    "capacitance_F": cap.capacitance_F,
                    "voltage_V": cap.voltage_V,
                    "charge_C": cap.charge_C,
                    "connected_to_source": cap.connected_to_source,
                }
                for cid, cap in system.capacitors.items()
            },
        )

    answer = _format_system_answer(problem.query, system)
    result.answer = answer
    result.add_step(
        "Final answer",
        f"The requested final {problem.query} is {answer}.",
    )

    return result


def _solve_single(question: str) -> SolveResult:
    result = SolveResult(status="solved", domain="capacitor_state")

    problem = extract_problem(question)
    state = problem.initial_state.clone()
    state.infer_missing()

    result.add_step(
        "Initial state",
        "Parsed the initial capacitor state.",
        capacitance_F=state.capacitance_F,
        voltage_V=state.voltage_V,
        charge_C=state.charge_C,
        connected_to_source=state.connected_to_source,
    )

    for event in problem.events:
        detail = event.apply(state)
        state.infer_missing()
        result.add_step(
            f"Apply event: {event.name}",
            detail,
            capacitance_F=state.capacitance_F,
            voltage_V=state.voltage_V,
            charge_C=state.charge_C,
            connected_to_source=state.connected_to_source,
        )

    answer = _format_single_answer(problem.query, state)
    result.answer = answer
    result.add_step(
        "Final answer",
        f"The requested final {problem.query} is {answer}.",
    )

    return result


def solve(question: str) -> SolveResult:
    try:
        parallel_result = _solve_parallel(question)
        if parallel_result is not None:
            return parallel_result

        return _solve_single(question)

    except Exception as exc:
        result = SolveResult(status="solve_failed", domain="capacitor_state")
        result.error = str(exc)
        return result
