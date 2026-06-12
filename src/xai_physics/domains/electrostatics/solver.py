from xai_physics.core.result import SolveResult


def solve(question: str) -> SolveResult:
    result = SolveResult(status="unsupported", domain="electrostatics")
    result.add_step(
        "Domain selected",
        "The router selected the geometry + electrostatics solver."
    )
    result.error = "Electrostatics geometry solver has not been migrated yet."
    return result
