from __future__ import annotations

import argparse
import json

from xai_physics.llm.schema_pipeline import solve_problem_with_llm
from xai_physics.llm.transformers_qwen_client import qwen_client_from_env


DEFAULT_PROBLEM = (
    "Calculate the energy stored in a capacitor with capacitance 100 uF and voltage 30 V."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", nargs="*", help="Problem text. If omitted, uses a capacitor-energy smoke test.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--show-raw", action="store_true")
    args = parser.parse_args()

    problem = " ".join(args.problem).strip() or DEFAULT_PROBLEM

    client = qwen_client_from_env()
    output = solve_problem_with_llm(problem, client, k=args.k)

    print("Status:", output.solve_result.status)
    print("Domain:", output.solve_result.domain)
    print("Answer:", output.solve_result.answer)
    print("Error:", output.solve_result.error)

    if output.schema is not None:
        print("\nSchema:")
        print(json.dumps(output.schema, ensure_ascii=False, indent=2))

    if args.show_raw:
        print("\nRaw LLM output:")
        print(output.raw_llm_output)

    if args.show_prompt:
        print("\nPrompt:")
        print(output.prompt_result.prompt)


if __name__ == "__main__":
    main()
