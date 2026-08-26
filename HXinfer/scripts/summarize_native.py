#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = (
    "completed",
    "duration",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result_path = args.run_dir / "benchmark.json"
    partition_path = args.run_dir / "partition.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    lines = [
        "# Native PP Run Summary",
        "",
        f"- Model: `{partition['model']}`",
        f"- Hidden layers: {partition['num_hidden_layers']}",
        f"- Expected stages: `{json.dumps(partition['stages'])}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name in METRICS:
        if name in result:
            lines.append(f"| {name} | {result[name]} |")
    lines.extend(
        [
            "",
            (
                "> This is an automatic extraction, not a PASS decision. Add actual "
                "worker layer ownership, memory, boundary tensor, stage timing and PP "
                "communication measurements before accepting the run."
            ),
            "",
        ]
    )
    (args.run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
