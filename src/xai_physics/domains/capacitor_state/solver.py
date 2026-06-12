from __future__ import annotations

from xai_physics.core.result import SolveResult
from xai_physics.domains.capacitor_state.extractor import extract_problem


def _format_answer(query: str, state) -> str:
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


def solve(question: str) -> SolveResult:
    result = SolveResult(status="solved", domain="capacitor_state")

    try:
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

        answer = _format_answer(problem.query, state)
        result.answer = answer
        result.add_step(
            "Final answer",
            f"The requested final {problem.query} is {answer}.",
        )

        return result

    except Exception as exc:
        result.status = "solve_failed"
        result.error = str(exc)
        return result
