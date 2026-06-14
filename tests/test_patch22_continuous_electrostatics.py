from xai_physics.eval.replay_llm_dataset import compare_answer
from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.schema_solver import solve_schema


QUESTIONS = {
    "DT072": "A thin circular ring with a radius R = 5 cm carries a total charge Q = 3 μC, uniformly distributed along its circumference. The ring lies in the xy-plane, centered at the origin. Calculate the magnitude of the electric field at point P located on the z-axis, 2.5 cm from the center.",
    "DT073": "A thin, non-conducting rod of length L = 10 cm is uniformly charged with a linear charge density of λ = 5 x 10^-6 C/m. The rod lies along the z-axis, from z = 0 to z = L. Calculate the net electric field at point M, which is located at a distance r = 6 cm from the origin (point O) along the x-axis.",
    "DT074": "Consider two wide, parallel insulating plates with surface charge densities of σ and (−σ), respectively. The distance between them is very small compared to the dimensions of the plates. What is the magnitude of the electric field at a point located between the two plates, given that σ = 8.85 × 10^-6 C/m^2?",
    "DT075": "Consider two wide parallel insulating sheets with identical surface charge densities of σ. The distance between them is very small compared to their dimensions. What is the magnitude of the electric field at a point located between the two sheets, given that σ = 8.85 × 10^-6 C/m^2?",
    "DT083": "A flat, circular conducting disk, with a radius R = 10 cm, is uniformly charged with a surface charge density σ. Calculate the axial component of the electric field strength, Ez, at point P located on the axis perpendicular to the center of the disk (i.e., the z-axis), at a distance z = 5 cm from the center of the disk, given that σ = 5 × 10^-6 C/m^2.",
    "DT089": "An infinitely large, flat metal plate is uniformly charged. It was determined that the charge contained on a 2 m x 5 m rectangular area is 4 µC. Calculate the electric field strength at point M, 20 cm away from the metal plate.",
    "DT091": "A charge Q = 0.7 x 10^^-9 C is uniformly distributed along a semicircle with center O and radius R = 20 cm. Determine the electric field strength at O.",
}

EXPECTED = {
    "DT072": ("3863925.47", "N/C"),
    "DT073": ("6.43 x 10^5", "N/C"),
    "DT074": ("1000000", "N/C"),
    "DT075": ("0", "N/C"),
    "DT083": ("156154.35", "V/m"),
    "DT089": ("2.26.10^4", "V/m"),
    "DT091": ("100", "V/m"),
}


def _continuous_candidate(case_id: str):
    candidates = [
        schema
        for schema in generate_equations_candidate_schemas(QUESTIONS[case_id])
        if schema.get("solver_backend") == "continuous_distribution"
    ]
    assert candidates, case_id
    return candidates[0]


def test_continuous_distribution_candidates_route_as_electrostatics_backend():
    schema = _continuous_candidate("DT072")

    assert schema["domain"] == "electrostatics"
    assert schema["representation"] == "numeric"
    assert schema["solver_backend"] == "continuous_distribution"
    result = solve_schema(schema)

    assert result.status == "solved"
    assert result.domain == "electrostatics"
    assert result.answer_meta is not None
    assert result.answer_meta.formula == "ring_axial_field"


def test_patch22_continuous_distribution_cases_match_expected_answers():
    for case_id, (expected, unit) in EXPECTED.items():
        result = solve_schema(_continuous_candidate(case_id))

        assert result.status == "solved", case_id
        assert compare_answer(result.answer, expected, expected_unit=unit) is True, (case_id, result.answer)
