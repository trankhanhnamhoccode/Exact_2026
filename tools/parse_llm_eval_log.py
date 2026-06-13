from __future__ import annotations

import argparse
from pathlib import Path

from xai_physics.llm.replay_cache import parse_eval_log, write_replay_cache


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a text LLM eval log into replay-cache JSONL.")
    parser.add_argument("log", type=Path, help="Text log from the LLM dataset benchmark.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSONL replay cache path.")
    args = parser.parse_args(argv)

    text = args.log.read_text(encoding="utf-8-sig", errors="replace")
    entries = parse_eval_log(text)
    count = write_replay_cache(entries, args.output)
    print(f"parsed {count} cached schema(s) -> {args.output}")


if __name__ == "__main__":
    main()
