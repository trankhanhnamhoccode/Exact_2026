from xai_physics.llm.domain_classifier import classify_domain
from xai_physics.llm.prompt_builder import build_schema_prompt


def test_classify_capacitor_state():
    q = (
        "A 500 pF capacitor is initially connected to a 300 V battery, "
        "then disconnected and a dielectric with constant 2 is inserted. "
        "Find the final voltage."
    )

    decision = classify_domain(q)

    assert decision.domain == "capacitor_state"
    assert "capacitor" in decision.tags


def test_classify_electrostatics():
    q = (
        "Three charges are placed at the vertices of an equilateral triangle "
        "of side 10 cm. Find the net force on q3."
    )

    decision = classify_domain(q)

    assert decision.domain == "electrostatics"
    assert "equilateral_triangle" in decision.tags or "geometry" in decision.tags


def test_build_prompt_uses_capacitor_prompt_only():
    q = "A capacitor is disconnected and a dielectric is inserted. Find the final voltage."

    built = build_schema_prompt(q)

    assert built.domain_decision.domain == "capacitor_state"
    assert "Domain:" in built.prompt
    assert "capacitor_state" in built.prompt
    assert "Electrostatics" not in built.prompt


def test_build_prompt_uses_electrostatics_prompt_only():
    q = "Two charges are placed 30 cm apart. Find the electric force on q2."

    built = build_schema_prompt(q)

    assert built.domain_decision.domain == "electrostatics"
    assert "electrostatics" in built.prompt
    assert "capacitor_state" not in built.prompt


def test_build_prompt_includes_retrieved_examples():
    q = "A capacitor is disconnected. Find voltage."

    examples = [
        {
            "problem": "A 500 pF capacitor is disconnected and k=2 dielectric is inserted.",
            "schema": {"domain": "capacitor_state"},
        }
    ]

    built = build_schema_prompt(q, examples=examples)

    assert "Retrieved examples" in built.prompt
    assert "500 pF" in built.prompt
