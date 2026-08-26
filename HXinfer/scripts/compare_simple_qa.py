#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare short QA correctness and tokens"
    )
    parser.add_argument("native", type=Path)
    parser.add_argument("external", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    native = json.loads(args.native.read_text(encoding="utf-8"))
    external = json.loads(args.external.read_text(encoding="utf-8"))
    native_answers = native["answers"]
    external_answers = external["answers"]
    if len(native_answers) != len(external_answers):
        raise SystemExit(
            f"answer counts differ: native={len(native_answers)}, "
            f"external={len(external_answers)}"
        )
    rows = []
    for index, (left, right) in enumerate(zip(native_answers, external_answers)):
        rows.append(
            {
                "index": index,
                "expected": left["expected"],
                "native": left["actual"],
                "external": right["actual"],
                "both_correct": left["correct"] and right["correct"],
                "token_ids_equal": left["output_token_ids"]
                == right["output_token_ids"],
            }
        )
    result = {
        "passed": all(row["both_correct"] and row["token_ids_equal"] for row in rows),
        "cases": rows,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
