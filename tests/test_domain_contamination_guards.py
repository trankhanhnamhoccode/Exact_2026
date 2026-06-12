from xai_physics.llm.domain_classifier import classify_domain
from xai_physics.llm.prompt_builder import build_schema_prompt


def test_capacitor_state_event_is_not_routed_to_equations():
    problem = (
        "A 500 pF capacitor is initially connected to a 300 V battery, "
        "then disconnected and a dielectric with constant 2 is inserted. "
        "Find the final voltage."
    )

    decision = classify_domain(problem)
    built = build_schema_prompt(problem)

    assert decision.domain == "capacitor_state"
    assert built.domain_decision.domain == "capacitor_state"
    assert "Domain:\ncapacitor_state" in built.prompt
    assert "Relevant formula docs:" not in built.prompt


def test_replace_dielectric_event_is_not_routed_to_equations():
    problem = (
        "A capacitor has dielectric constant 4 and it is replaced by dielectric constant 2. "
        "Find the ratio of final capacitance to initial capacitance."
    )

    decision = classify_domain(problem)
    built = build_schema_prompt(problem)

    assert decision.domain == "capacitor_state"
    assert built.domain_decision.domain == "capacitor_state"
    assert "Domain:\ncapacitor_state" in built.prompt
    assert "Relevant formula docs:" not in built.prompt


def test_electrostatics_geometry_is_not_routed_to_equations():
    problem = (
        "Three charges are placed at the vertices of an equilateral triangle. "
        "Find the net electrostatic force on charge q1."
    )

    decision = classify_domain(problem)
    built = build_schema_prompt(problem)

    assert decision.domain == "electrostatics"
    assert built.domain_decision.domain == "electrostatics"
    assert "Domain:\nelectrostatics" in built.prompt
    assert "Relevant formula docs:" not in built.prompt


def test_plain_capacitor_formula_is_routed_to_equations():
    problem = "Calculate the energy stored in a capacitor with capacitance 100 uF and voltage 30 V."

    decision = classify_domain(problem)
    built = build_schema_prompt(problem)

    assert decision.domain == "equations"
    assert built.domain_decision.domain == "equations"
    assert "Domain:\nequations" in built.prompt
    assert "Relevant formula docs:" in built.prompt
    assert built.formulas[0]["id"] == "capacitor_energy_voltage"


def test_plain_point_charge_field_scalar_can_be_equations():
    problem = "Find the electric field at a distance of 30 cm from a point charge of 2 uC."

    decision = classify_domain(problem)
    built = build_schema_prompt(problem)

    assert decision.domain == "equations"
    assert built.domain_decision.domain == "equations"
    assert built.formulas[0]["id"] == "point_charge_electric_field"


def test_point_charges_with_geometry_go_to_electrostatics():
    problem = (
        "Two point charges are placed at points A and B on a straight line. "
        "Find the net force on the charge at A."
    )

    decision = classify_domain(problem)
    built = build_schema_prompt(problem)

    assert decision.domain == "electrostatics"
    assert built.domain_decision.domain == "electrostatics"
    assert "Domain:\nelectrostatics" in built.prompt
