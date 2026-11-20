"""Bounded-memory stage timing summaries."""

from random import Random

import numpy as np


class Timings:
    def __init__(self, sample_capacity: int = 8192):
        self.capacity = sample_capacity
        self.random = Random(42)
        self.samples: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}
        self.totals: dict[str, float] = {}

    def record(self, stage: str, seconds: float) -> None:
        count = self.counts.get(stage, 0) + 1
        self.counts[stage] = count
        self.totals[stage] = self.totals.get(stage, 0.0) + seconds
        sample = self.samples.setdefault(stage, [])
        if count <= self.capacity:
            sample.append(seconds)
        else:
            index = self.random.randrange(count)
            if index < self.capacity:
                sample[index] = seconds

    def summary(self) -> dict[str, dict[str, float | int]]:
        result = {}
        for stage, count in self.counts.items():
            p50, p95, p99 = np.percentile(self.samples[stage], [50, 95, 99]) * 1000
            result[stage] = {
                "count": count,
                "mean": self.totals[stage] * 1000 / count,
                "p50": float(p50),
                "p95": float(p95),
                "p99": float(p99),
            }
        return result
