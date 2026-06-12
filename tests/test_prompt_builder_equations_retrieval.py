from xai_physics.llm.prompt_builder import build_schema_prompt


def test_equations_prompt_injects_retrieved_formula_docs_not_full_store():
    result = build_schema_prompt(
        "Calculate the energy stored in a capacitor with capacitance 100 uF and voltage 30 V.",
        k=2,
    )

    assert result.domain_decision.domain == "equations"
    assert result.formulas[0]["id"] == "capacitor_energy_voltage"

    prompt = result.prompt

    assert "Domain:\nequations" in prompt
    assert "Relevant formula docs:" in prompt
    assert "capacitor_energy_voltage" in prompt
    assert "W = 1/2 C U^2" in prompt
    assert "Schema template:" in prompt
    assert "Retrieved examples:" in prompt
    assert "eq_ex_cap_energy_001" in prompt

    # Không dump full formula store vào prompt.
    assert "harmonic_current_cos_time" not in prompt
    assert "solenoid_magnetic_field" not in prompt


def test_equations_prompt_retrieves_lc_formula_docs():
    result = build_schema_prompt(
        "Find the resonant frequency of an LC circuit with inductance 0.5 H and capacitance 5 uF.",
        k=2,
    )

    assert result.domain_decision.domain == "equations"
    assert result.formulas[0]["id"] == "lc_resonance_frequency"
    assert "lc_resonance_frequency" in result.prompt
    assert "f = 1/(2*pi*sqrt(L*C))" in result.prompt
    assert "eq_ex_lc_freq_001" in result.prompt


def test_equations_prompt_retrieval_debug_is_exposed():
    result = build_schema_prompt(
        "Find the current when voltage is 120 V and resistance is 40 ohm.",
        k=2,
    )

    assert result.domain_decision.domain == "equations"
    assert result.formulas[0]["id"] == "ohm_law"
    assert "formula_rule_scores" in result.retrieval_debug
    assert "tag_hits" in result.retrieval_debug
