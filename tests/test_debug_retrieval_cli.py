from xai_physics.cli.debug_retrieval import build_debug_report


def test_debug_retrieval_report_for_capacitor_problem():
    problem = (
        "An air-filled parallel-plate capacitor has C = 500 pF and U = 300 V. "
        "It is disconnected from the source and immersed in a dielectric with epsilon = 2. "
        "What is the potential difference?"
    )

    report = build_debug_report(problem)

    assert "Retrieval Debug Report" in report
    assert "capacitor_state" in report
    assert "isolated_dielectric" in report
    assert "cap_td001" in report


def test_debug_retrieval_report_can_show_prompt():
    problem = (
        "Two capacitors C1 = 4 uF and C2 = 6 uF are charged to 150 V and 300 V. "
        "After connecting their like-poled terminals together, find the final voltage."
    )

    report = build_debug_report(problem, show_prompt=True, max_prompt_chars=20000)

    assert "ParallelRedistribution" in report
    assert "cap_td095" in report
    assert "Return JSON only" in report

