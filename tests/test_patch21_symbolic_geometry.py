from xai_physics.eval.replay_llm_dataset import compare_answer
from xai_physics.hybrid.equations_candidates import generate_equations_candidate_schemas
from xai_physics.schema_solver import solve_schema


_QUESTIONS = {
    'DT007': 'Two electric charges q1 = q2 = q (with q > 0) are placed at two points A and B, with the distance AB = 2a (m). Point M is located on the perpendicular bisector of the line segment AB, at a distance h from AB. Determine the value of h for which the electric field strength at M is maximum, and calculate this value of h.',
    'DT008': 'Two charges, q1 = q2 = q (where q > 0, in Coulombs), are placed at points A and B, with the distance AB = 2a (meters). Point M is located on the perpendicular bisector of the line segment AB, at a distance h from AB. Determine the magnitude of the electric field vector at point M. Given k = 9 × 10^9.',
    'DT019': "Place four charges of the same magnitude q at the four vertices of a square ABCD with side length a. Positive charges are placed at A and C, and negative charges are placed at B and D. Determine the net electric field at the intersection of the square's two diagonals.",
    'DT020': 'Four charges of the same magnitude q (C) are placed at the four vertices of a square ABCD with side length a (m). Positive charges are placed at vertices A and D, and negative charges are placed at vertices B and C. Determine the resultant electric field strength at the intersection point of the two diagonals of the square.',
    'DT040': "At two vertices A and B of an equilateral triangle ABC with side length 'a', two point charges q1 = q2 = 4 x 10^-9 C are placed in air. What value must charge q3 have at vertex C so that the electric field strength produced by the system of three charges at the centroid G of the triangle is zero?",
    'DT047': 'A charge q is placed at point O in the air. Ox is an electric field line. Take two points A and B on Ox. Let M be the midpoint of AB. E_A is the electric field strength at A, and E_B is the electric field strength at B. Determine 1/sqrt(E_M) in terms of E_A and E_B.',
}


def _question(case_id: str) -> str:
    return _QUESTIONS[case_id]


def _symbolic_candidates(case_id: str):
    return [
        schema
        for schema in generate_equations_candidate_schemas(_question(case_id))
        if schema.get('representation') == 'symbolic' and schema.get('solver_backend') in {'symbolic_geometry', 'symbolic_relation'}
    ]


def test_symbolic_perpendicular_bisector_field_expression_dt008():
    schema = _symbolic_candidates('DT008')[0]
    assert schema['domain'] == 'electrostatics'
    assert schema['representation'] == 'symbolic'
    assert schema['solver_backend'] == 'symbolic_geometry'
    assert schema['queries'][0]['answer_mode'] == 'symbolic_expr'
    result = solve_schema(schema)

    assert result.status == 'solved'
    assert result.domain == 'electrostatics'
    assert result.answer_meta is not None
    assert compare_answer(result.answer, r'/frac{2k \abs{q} h}{(a^2 + h^2)^1.5}', expected_unit='V/m') is True


def test_symbolic_perpendicular_bisector_maximum_dt007():
    schema = _symbolic_candidates('DT007')[0]
    result = solve_schema(schema)

    assert result.status == 'solved'
    assert compare_answer(result.answer, r'a/ \sqrt{2}', expected_unit='m') is True


def test_symbolic_square_sign_pattern_zero_and_nonzero():
    zero_schema = _symbolic_candidates('DT019')[0]
    zero_result = solve_schema(zero_schema)
    assert zero_result.status == 'solved'
    assert compare_answer(zero_result.answer, '0', expected_unit='V/m') is True

    nonzero_schema = _symbolic_candidates('DT020')[0]
    nonzero_result = solve_schema(nonzero_schema)
    assert nonzero_result.status == 'solved'
    assert compare_answer(nonzero_result.answer, r'\frac{4 \sqrt{2} k q}{\epsilon a^2}', expected_unit='V/m') is True


def test_symbolic_equilateral_centroid_unknown_charge_dt040():
    schema = _symbolic_candidates('DT040')[0]
    result = solve_schema(schema)

    assert result.status == 'solved'
    assert compare_answer(result.answer, r'4 . 10^{-9}', expected_unit='C') is True


def test_symbolic_midpoint_inverse_sqrt_relation_dt047():
    schema = _symbolic_candidates('DT047')[0]
    result = solve_schema(schema)

    assert result.status == 'solved'
    assert result.answer_meta is not None
    assert compare_answer(result.answer, r'1/2 . (1/ \sqrt{E_A} + 1/ \sqrt{E_B})') is True


def test_legacy_symbolic_geometry_domain_still_routes():
    schema = dict(_symbolic_candidates('DT008')[0])
    schema['domain'] = 'symbolic_geometry'
    schema.pop('representation', None)
    schema.pop('solver_backend', None)
    result = solve_schema(schema)

    assert result.status == 'solved'
    assert result.domain == 'symbolic_geometry'
