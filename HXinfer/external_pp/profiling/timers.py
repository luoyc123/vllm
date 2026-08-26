from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter_ns


class PhaseTimer:
    """Collect non-overlapping wall-clock phase durations in milliseconds."""

    def __init__(self) -> None:
        self.samples_ms: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        start = perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter_ns() - start) / 1_000_000
            self.samples_ms.setdefault(phase, []).append(elapsed_ms)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            phase: {
                "count": float(len(values)),
                "total_ms": sum(values),
                "mean_ms": sum(values) / len(values),
            }
            for phase, values in self.samples_ms.items()
        }
