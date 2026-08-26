#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: Path, event: str) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit(f"no records in {path}")
    unexpected = [record for record in records if record.get("event") != event]
    if unexpected:
        raise SystemExit(f"unexpected events in {path}: {unexpected[0]}")
    return records


def throughput_gbps(byte_count: int, elapsed_ns: int) -> float:
    if elapsed_ns <= 0:
        raise ValueError("elapsed time must be positive")
    return byte_count / elapsed_ns


def print_stage(
    label: str,
    byte_count: int,
    elapsed_ns: int,
    pcie_gbps: float,
) -> None:
    gbps = throughput_gbps(byte_count, elapsed_ns)
    print(
        f"{label}: bytes={byte_count} time_ms={elapsed_ns / 1e6:.3f} "
        f"effective_GBps={gbps:.3f} pcie_theoretical_pct={gbps / pcie_gbps * 100:.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze HXinfer activation transport JSONL logs"
    )
    parser.add_argument("rank0", type=Path, help="transport-rank0.jsonl")
    parser.add_argument("rank1", type=Path, help="transport-rank1.jsonl")
    parser.add_argument(
        "--pcie-gbps",
        type=float,
        default=31.507692,
        help="Theoretical one-way PCIe payload GB/s (default: PCIe 4.0 x16)",
    )
    args = parser.parse_args()

    sends = load_records(args.rank0, "send")
    receives = load_records(args.rank1, "recv")
    send_pairs = [(record["sequence"], record["bytes"]) for record in sends]
    receive_pairs = [(record["sequence"], record["bytes"]) for record in receives]
    if send_pairs != receive_pairs:
        raise SystemExit("rank0/rank1 sequence or payload size mismatch")

    total_bytes = sum(record["bytes"] for record in sends)
    largest_index = max(range(len(sends)), key=lambda index: sends[index]["bytes"])
    largest_send = sends[largest_index]
    largest_recv = receives[largest_index]

    print(f"messages={len(sends)} payload_bytes={total_bytes} pair_check=PASS")
    print_stage(
        "aggregate CUDA D2H",
        total_bytes,
        sum(record["d2h_ns"] for record in sends),
        args.pcie_gbps,
    )
    print_stage(
        "aggregate sender sendall",
        total_bytes,
        sum(record["socket_ns"] for record in sends),
        args.pcie_gbps,
    )
    print_stage(
        "aggregate ROCm H2D",
        total_bytes,
        sum(record["h2d_ns"] for record in receives),
        args.pcie_gbps,
    )
    print(f"largest sequence={largest_send['sequence']} bytes={largest_send['bytes']}")
    print_stage(
        "largest CUDA D2H",
        largest_send["bytes"],
        largest_send["d2h_ns"],
        args.pcie_gbps,
    )
    print_stage(
        "largest sender sendall",
        largest_send["bytes"],
        largest_send["socket_ns"],
        args.pcie_gbps,
    )
    print_stage(
        "largest ROCm H2D",
        largest_recv["bytes"],
        largest_recv["h2d_ns"],
        args.pcie_gbps,
    )
    print(
        "note: these are software-stage effective throughputs, not raw PCIe or "
        "end-to-end transport bandwidth"
    )


if __name__ == "__main__":
    main()
