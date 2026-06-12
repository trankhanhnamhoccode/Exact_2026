from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from xai_physics.llm.prompt_builder import build_schema_prompt


def _fmt_list(items: list[str]) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(items) + "]"


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... <truncated> ..."


def build_debug_report(
    problem: str,
    show_prompt: bool = False,
    max_prompt_chars: int = 2000,
) -> str:
    built = build_schema_prompt(problem)

    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("Retrieval Debug Report")
    lines.append("=" * 80)

    lines.append("")
    lines.append("[Problem]")
    lines.append(problem)

    lines.append("")
    lines.append("[Domain decision]")
    lines.append(f"domain     : {built.domain_decision.domain}")
    lines.append(f"confidence : {built.domain_decision.confidence:.4f}")
    lines.append(f"reason     : {built.domain_decision.reason}")
    lines.append(f"classifier tags : {_fmt_list(built.domain_decision.tags)}")

    lines.append("")
    lines.append("[Final tags]")
    lines.append(_fmt_list(built.tags))

    lines.append("")
    lines.append("[Selected examples]")
    if not built.examples:
        lines.append("(none)")
    else:
        for i, ex in enumerate(built.examples, 1):
            retrieval: dict[str, Any] = ex.get("retrieval", {})
            lines.append(f"{i}. {ex.get('id')}")
            lines.append(f"   tags         : {_fmt_list(ex.get('tags', []))}")
            lines.append(f"   problem      : {ex.get('problem')}")
            if retrieval:
                lines.append(f"   vector_score : {retrieval.get('vector_score')}")
                lines.append(f"   rule_score   : {retrieval.get('rule_score')}")
                lines.append(f"   rerank_score : {retrieval.get('rerank_score')}")
                lines.append(f"   matched_tags : {_fmt_list(retrieval.get('matched_tags', []))}")

    vector_candidates = built.retrieval_debug.get("vector_candidates", [])

    lines.append("")
    lines.append("[Vector candidates]")
    if not vector_candidates:
        lines.append("(none)")
    else:
        for cand in vector_candidates:
            lines.append(
                f"- {cand.get('id')} | score={cand.get('score')} | tags={cand.get('tags')}"
            )

    lines.append("")
    lines.append("[Prompt]")
    if show_prompt:
        lines.append(_clip(built.prompt, max_prompt_chars))
    else:
        lines.append("hidden; pass --show-prompt to display it")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, help="Raw physics problem text.")
    parser.add_argument("--file", type=Path, help="Read problem text from a file.")
    parser.add_argument("--show-prompt", action="store_true", help="Display prompt preview.")
    parser.add_argument("--max-prompt-chars", type=int, default=2000)

    args = parser.parse_args()

    if args.file:
        problem = args.file.read_text(encoding="utf-8-sig").strip()
    elif args.problem:
        problem = args.problem.strip()
    else:
        raise SystemExit("Provide either --problem or --file.")

    print(
        build_debug_report(
            problem=problem,
            show_prompt=args.show_prompt,
            max_prompt_chars=args.max_prompt_chars,
        )
    )


if __name__ == "__main__":
    main()
