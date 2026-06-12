from xai_physics.core.result import SolveResult


def solve(question: str) -> SolveResult:
    result = SolveResult(status="unsupported", domain="equations")
    result.add_step(
        "Domain selected",
        "The router selected the scalar equation solver."
    )
    result.error = "Equation solver has not been migrated yet."
    return result
