from __future__ import annotations

import json

from xai_physics.eval.replay_llm_dataset import compare_answer
from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.llm.schema_pipeline import solve_problem_with_llm
from xai_physics.schema_solver import solve_schema


class _FakeClient:
    def __init__(self, schema: dict | None = None):
        self.schema = schema or {"domain": "equations", "objects": [], "relations": []}

    def generate(self, prompt: str) -> str:
        return json.dumps(self.schema)


def _solve_formula(question: str, formula_name: str):
    for schema in generate_equations_candidate_schemas(question):
        names = [rel.get("name") for rel in schema.get("relations", []) if isinstance(rel, dict)]
        if formula_name in names:
            return solve_schema(schema)
    raise AssertionError(f"formula candidate not generated: {formula_name}")


def test_electromechanics_formula_candidates():
    electron = (
        "An electron moves along the electric field lines of a uniform electric field with an electric field strength "
        "E = 100 V / m. Its initial velocity is 300 km / s, in the direction of the electric field vector E. "
        "What distance does the electron travel before its velocity reduces to zero?"
    )
    result = _solve_formula(electron, "electron_stopping_distance_uniform_field")
    assert result.status == "ok"
    assert compare_answer(result.answer, "2.56", expected_unit="mm") is True

    dust = (
        "The electric field between two vertical, oppositely charged metal plates has a magnitude of 4900 V/m. "
        "Determine the mass of a dust particle placed in this electric field if it carries a charge q = 4.10^-10 C "
        "and is in equilibrium, with its suspension thread making an angle of 30° with the vertical."
    )
    result = _solve_formula(dust, "charged_dust_equilibrium_mass")
    assert result.status == "ok"
    assert compare_answer(result.answer, "3.4 . 10^{-7}", expected_unit="kg") is True


def test_inverse_rectangle_and_numeric_geometry_candidates():
    q1 = (
        "Four points A, B, C, D in the air form a rectangle ABCD with sides AD = 3 cm and AB = 4 cm. "
        "Charges q1, q2, q3 are placed respectively at A, B, C. Let E2 be the electric field vector created by q2 at D, "
        "and E13 be the resultant electric field vector created by charges q1 and q3 at D. Determine the value of q1, "
        "given that q2 = −12.5 x 10⁻⁸ C and E2 = E13 (as vectors)."
    )
    result = _solve_formula(q1, "rectangle_inverse_field_charge")
    assert compare_answer(result.answer, "-2.7 . 10^{-8}", expected_unit="C") is True

    q3 = q1.replace("value of q1", "value of q3")
    result = _solve_formula(q3, "rectangle_inverse_field_charge")
    assert compare_answer(result.answer, "-6.4 . 10^{-8}", expected_unit="C") is True

    square = (
        "At three vertices of a square with side length a = 40cm, three equal positive point charges "
        "(q1 = q2 = q3 = 5 x 10^-9 C) are placed. What is the magnitude of the electric field vector "
        "at the fourth vertex of the square?"
    )
    result = _solve_formula(square, "two_charge_geometry_field")
    assert compare_answer(result.answer, "538", expected_unit="V/m") is True

    bisector = (
        "Two point charges q1 = -3.6 μC and q2 = -2.1 μC are placed at points A and B, 6 cm apart. "
        "Find the electric field at point M, which is equidistant from A and B, lies on the line perpendicular to AB "
        "through the midpoint of AB, and is 5 cm from the midpoint."
    )
    result = _solve_formula(bisector, "two_charge_geometry_field")
    assert compare_answer(result.answer, "1.31*10^7", expected_unit="V/m") is True


def test_capacitor_state_transition_formula_candidates():
    series = (
        "A capacitor C = 5 μF is charged to 20 V. It is then connected in series with an uncharged capacitor C'. "
        "The entire circuit has a total voltage of 20 V. If the final charge on C is 30 μC, find C'."
    )
    result = _solve_formula(series, "series_uncharged_capacitor_from_final_charge")
    assert compare_answer(result.answer, "2.140", expected_unit="μF") is True

    sharing = (
        "A 2 μF capacitor is charged to 12 V, then disconnected, and its charge is equally shared among two identical capacitors. "
        "Calculate the total remaining energy after sharing."
    )
    result = _solve_formula(sharing, "identical_capacitor_charge_sharing_energy")
    assert compare_answer(result.answer, "72", expected_unit="μJ") is True

    series_energy = (
        "A capacitor has an energy of 2 mJ and a capacitance of 1 μF. If it is connected in series with another 1 μF "
        "uncharged capacitor, calculate the new total energy of the system."
    )
    result = _solve_formula(series_energy, "energy_shared_equal_capacitor_series")
    assert compare_answer(result.answer, "1", expected_unit="mJ") is True

    dielectric = (
        "A parallel-plate capacitor is fully charged and then disconnected from its power source. Subsequently, it is placed "
        "in an environment where the permittivity (ε) increases by a factor of 3. Calculate the new energy stored in the capacitor "
        "if the initial energy was 1 μJ."
    )
    result = _solve_formula(dielectric, "disconnected_dielectric_energy_scaling")
    assert compare_answer(result.answer, "0.33", expected_unit="μJ") is True

    split_three = (
        "A capacitor with capacitance C = 12 μF is charged to U = 100 V. If its charge is then distributed equally among 3 "
        "identical capacitors, what is the total energy of the system?"
    )
    result = _solve_formula(split_three, "identical_capacitor_charge_sharing_energy")
    assert compare_answer(result.answer, "0.020", expected_unit="J") is True


def test_direct_point_charge_candidate_respects_force_and_dielectric_contexts():
    # Force-aware field should use F/q, not the direct point-charge fallback.
    field_from_force = (
        "A charge q = 10^-7 C is placed in the electric field of a point charge Q, and experiences a force F = 3 mN. "
        "Calculate the electric field strength at the point where charge q is placed, given that the two charges are separated "
        "by a distance of 30 cm in a vacuum."
    )
    pipeline = solve_problem_with_llm(field_from_force, _FakeClient())
    assert compare_answer(pipeline.solve_result.answer, "3 . 10^4", expected_unit="V/m") is True

    charge_from_force = field_from_force.replace(
        "Calculate the electric field strength at the point where charge q is placed",
        "Calculate the magnitude of charge Q",
    )
    pipeline = solve_problem_with_llm(charge_from_force, _FakeClient())
    assert compare_answer(pipeline.solve_result.answer, "3 . 10^{-7}", expected_unit="C") is True

    dielectric = (
        "A point charge q = 80 nC is fixed at O in oil. The dielectric constant of the oil is ε = 4. "
        "What is the electric field strength produced by q at point M, which is at a distance MO = 30 cm from O?"
    )
    result = _solve_formula(dielectric, "point_charge_electric_field")
    assert compare_answer(result.answer, "2000", expected_unit="V/m") is True


def test_hybrid_selection_for_multi_output_and_capacitor_transition():
    multi = (
        "At two points A and B, separated by 20 cm in air, two point charges q1 = 4 × 10^-6 C and q2 = -6.4 × 10^-6 C are placed. "
        "Determine the electric field strength caused by these two point charges at point C, knowing that AC = 12 cm and BC = 16 cm. "
        "Determine the electric force acting on charge q3 = -5 × 10^-8 C placed at C."
    )
    result = solve_problem_with_llm(multi, _FakeClient()).solve_result
    assert isinstance(result.answer, dict)
    assert compare_answer(result.answer, "33.6 × 10^5", expected_unit="V/m") is True

    cap_transition = (
        "A capacitor C = 5 μF is charged to 20 V. It is then connected in series with an uncharged capacitor C'. "
        "The entire circuit has a total voltage of 20 V. If the final charge on C is 30 μC, find C'."
    )
    result = solve_problem_with_llm(cap_transition, _FakeClient()).solve_result
    assert compare_answer(result.answer, "2.140", expected_unit="μF") is True
