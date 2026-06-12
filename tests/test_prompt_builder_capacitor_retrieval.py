from xai_physics.llm.prompt_builder import build_schema_prompt


def test_capacitor_prompt_uses_domain_retrieval_base():
    problem = (
        "An air-filled parallel-plate capacitor has C = 500 pF and U = 300 V. "
        "It is disconnected from the source and immersed in a dielectric with epsilon = 2. "
        "What is the potential difference?"
    )

    built = build_schema_prompt(problem)

    assert built.domain_decision.domain == "capacitor_state"
    assert "isolated_dielectric" in built.tags
    assert built.examples
    assert built.examples[0]["id"] == "cap_td001"
    assert "Retrieved examples" in built.prompt
    assert "capacitance" in built.prompt
    assert "DisconnectFromSource" in built.prompt
    assert "InsertDielectric" in built.prompt


def test_capacitor_prompt_parallel_like_polarity_retrieval():
    problem = (
        "Two capacitors C1 = 4 uF and C2 = 6 uF are charged to 150 V and 300 V. "
        "After connecting their like-poled terminals together, find the final voltage."
    )

    built = build_schema_prompt(problem)

    assert built.domain_decision.domain == "capacitor_state"
    assert "like_polarity_connection" in built.tags
    assert "charge_redistribution" in built.tags
    assert built.examples
    assert built.examples[0]["id"] == "cap_td095"
    assert "ParallelRedistribution" in built.prompt


def test_electrostatics_prompt_does_not_use_capacitor_retrieval():
    problem = (
        "Three charges are placed at the vertices of an equilateral triangle "
        "of side 10 cm. Find the net force on q3."
    )

    built = build_schema_prompt(problem)

    assert built.domain_decision.domain == "electrostatics"
    assert built.examples
    assert all(ex.get("domain") == "electrostatics" for ex in built.examples)
    assert "cap_td" not in built.prompt
