from __future__ import annotations

from xai_physics.router.domain_router import route_domain
from xai_physics.domains.equations import solver as equation_solver
from xai_physics.domains.electrostatics import solver as electrostatics_solver
from xai_physics.domains.capacitor_state import solver as capacitor_state_solver


def solve(question: str):
    domain = route_domain(question)

    if domain == "equations":
        return equation_solver.solve(question)

    if domain == "electrostatics":
        return electrostatics_solver.solve(question)

    if domain == "capacitor_state":
        return capacitor_state_solver.solve(question)

    raise ValueError(f"Unknown domain: {domain}")


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = "A capacitor is initially connected to a 300 V battery, then disconnected and a dielectric is inserted. Find the final voltage."

    result = solve(question)

    print("Status:", result.status)
    print("Domain:", result.domain)
    print("Answer:", result.answer)
    print("Error:", result.error)

    print("\nTrace:")
    for i, step in enumerate(result.trace, 1):
        print(f"{i}. {step.title}: {step.detail}")
