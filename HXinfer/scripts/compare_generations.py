#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Native and External QA outputs"
    )
    parser.add_argument("native", type=Path)
    parser.add_argument("external", type=Path)
    args = parser.parse_args()
    native = json.loads(args.native.read_text(encoding="utf-8"))
    external = json.loads(args.external.read_text(encoding="utf-8"))
    native_questions = native["questions"]
    external_questions = external["questions"]
    if len(native_questions) != len(external_questions):
        raise SystemExit("question counts differ")
    rows = []
    passed = True
    for index, (left, right) in enumerate(zip(native_questions, external_questions)):
        token_equal = left["output_token_ids"] == right["output_token_ids"]
        text_equal = left["text"] == right["text"]
        passed &= token_equal and text_equal
        rows.append(
            {"index": index, "token_ids_equal": token_equal, "text_equal": text_equal}
        )
    print(json.dumps({"passed": passed, "questions": rows}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
