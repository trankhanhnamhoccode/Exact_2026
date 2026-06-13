from __future__ import annotations

import os

import pytest

from xai_physics.llm.schema_pipeline import solve_problem_with_llm
from xai_physics.llm.transformers_qwen_client import qwen_client_from_env


pytestmark = pytest.mark.skipif(
    os.environ.get("XAI_RUN_QWEN_TESTS") != "1",
    reason="Set XAI_RUN_QWEN_TESTS=1 to run Kaggle/Transformers Qwen integration tests.",
)


def test_transformers_qwen_schema_pipeline_smoke():
    problem = "Calculate the energy stored in a capacitor with capacitance 100 uF and voltage 30 V."

    client = qwen_client_from_env()
    output = solve_problem_with_llm(problem, client, k=2)

    assert output.schema is not None, output.raw_llm_output
    assert output.schema.get("domain") == "equations"
    assert output.solve_result.status == "ok", output.solve_result.error
    assert output.solve_result.answer == "0.045 J"
