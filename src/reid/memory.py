"""Bounded global identity registry and dormant-candidate similarity."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from math import exp, hypot, isfinite, log
from typing import Mapping

import numpy as np

from src.types import Detection

from .appearance import Appearance, similarity


DEFAULT_WEIGHTS = {"appearance": 0.50, "motion": 0.22, "size": 0.13, "edge": 0.10, "time": 0.05}


class IdentityState(Enum):
    ACTIVE = auto()
    LOST = auto()
    INACTIVE = auto()


def weights_with_defaults(overrides: Mapping[str, float] | None) -> dict[str, float]:
    weights = DEFAULT_WEIGHTS | dict(overrides or {})
    if weights.keys() - DEFAULT_WEIGHTS.keys():
        raise ValueError("Unknown Re-ID weight")
    if any(not isfinite(weight) or weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Re-ID weights must be finite, nonnegative, with a positive sum")
    return weights


@dataclass(slots=True)
class Identity:
    id: int
    first_frame: int
    observation: Detection
    last_frame: int
    last_seen: float
    velocity: tuple[float, float]
    size: tuple[float, float]
    state: IdentityState = IdentityState.ACTIVE
    samples: deque[Appearance] = field(default_factory=lambda: deque(maxlen=4))
    recent: deque[tuple[int, Detection]] = field(default_factory=lambda: deque(maxlen=8))
    appearance_at: float = float("-inf")

    def record(
        self, detection: Detection, frame_index: int, timestamp: float, velocity: tuple[float, float]
    ) -> None:
        x1, y1, x2, y2 = detection.box
        self.size = (self.size[0] * 0.8 + (x2 - x1) * 0.2, self.size[1] * 0.8 + (y2 - y1) * 0.2)
        self.observation = detection
        self.last_frame = frame_index
        self.last_seen = timestamp
        self.velocity = velocity
        self.recent.append((frame_index, detection))

    def remember(self, appearance: Appearance | None, timestamp: float) -> None:
        self.appearance_at = timestamp
        if appearance is None:
            return
        if len(self.samples) == self.samples.maxlen:
            worst = min(range(len(self.samples)), key=lambda index: self.samples[index].quality)
            if appearance.quality + 0.05 < self.samples[worst].quality:
                return
            del self.samples[worst]
        self.samples.append(appearance)


class IdentityMemory:
    def __init__(self, max_seconds: float, max_inactive: int, weights: Mapping[str, float] | None) -> None:
        self.records: dict[int, Identity] = {}
        self.max_seconds = max_seconds
        self.max_inactive = max_inactive
        self.weights = weights_with_defaults(weights)
        self.next_id = 1

    def create(self, detection: Detection, frame_index: int, timestamp: float) -> Identity:
        x1, y1, x2, y2 = detection.box
        identity = Identity(self.next_id, frame_index, detection, frame_index, timestamp, (0.0, 0.0), (x2 - x1, y2 - y1))
        identity.recent.append((frame_index, detection))
        self.records[identity.id] = identity
        self.next_id += 1
        return identity

    def expire(self, timestamp: float) -> None:
        inactive = [record for record in self.records.values() if record.state is IdentityState.INACTIVE]
        # Oldest dormant identities are evicted first. Motion state stays separate.
        inactive.sort(key=lambda record: record.last_seen, reverse=True)
        for index, record in enumerate(inactive):
            if index >= self.max_inactive or timestamp - record.last_seen > self.max_seconds:
                del self.records[record.id]

    def scores(
        self,
        candidates: list[Identity],
        detections: list[Detection],
        appearances: list[Appearance | None],
        timestamp: float,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        scores = np.full((len(candidates), len(detections)), -1.0)
        for row, record in enumerate(candidates):
            for col, (detection, appearance) in enumerate(zip(detections, appearances)):
                scores[row, col] = self._similarity(record, detection, appearance, timestamp, image_size)
        return scores

    def _similarity(
        self,
        record: Identity,
        new: Detection,
        appearance: Appearance | None,
        timestamp: float,
        image_size: tuple[int, int],
    ) -> float:
        width, height = image_size
        age = max(0.0, timestamp - record.last_seen)
        left, top, right, bottom = record.observation.box
        old_x, old_y = (left + right) / 2, (top + bottom) / 2
        left, top, right, bottom = new.box
        new_x, new_y = (left + right) / 2, (top + bottom) / 2
        size = exp(-0.65 * (
            abs(log((right - left) / record.size[0])) + abs(log((bottom - top) / record.size[1]))
        ))
        if size < 0.36:
            return -1.0

        # Extrapolate for only the initial part of a gap, then let position evidence fade.
        vx, vy = record.velocity
        projection = min(age, 0.45)
        predicted_x = min(width, max(0.0, old_x + vx * projection))
        predicted_y = min(height, max(0.0, old_y + vy * projection))
        distance = hypot(new_x - predicted_x, new_y - predicted_y)
        tolerance = 12 + 0.8 * hypot(*record.size) + 0.033 * hypot(width, height) * (1 + age)
        proximity = exp(-0.5 * (distance / tolerance) ** 2)
        motion_weight = self.weights["motion"] / (1 + age * 0.5)

        old_edges = (old_x / width, old_y / height, 1 - old_x / width, 1 - old_y / height)
        new_edges = (new_x / width, new_y / height, 1 - new_x / width, 1 - new_y / height)
        exit_edge = min(range(4), key=old_edges.__getitem__)
        edge_known = old_edges[exit_edge] <= 0.12
        edge = max(0.0, 1 - 4 * new_edges[exit_edge])
        edge_weight = self.weights["edge"] if edge_known else 0.0
        same_edge = edge_known and new_edges[exit_edge] <= 0.12

        appearance_score, quality = 0.0, 0.0
        if appearance is not None and record.samples:
            matches = [(similarity(appearance, sample), min(appearance.quality, sample.quality)) for sample in record.samples]
            appearance_score, quality = max(matches, key=lambda match: match[0] * match[1])
        if quality >= 0.55 and appearance_score < 0.47:
            return -1.0
        if (quality < 0.45 or appearance_score < 0.68) and (
            age > 3.0 or proximity < (0.68 if same_edge else 0.87)
        ):
            # For tiny crops, allow a longer gap only when the motion points out of the
            # same edge and the next box returns very close along that boundary.
            outward_speed = (-vx, -vy, vx, vy)[exit_edge]
            lateral_offset, edge_length = (
                (abs(new_y - old_y), height) if exit_edge in (0, 2) else (abs(new_x - old_x), width)
            )
            plausible_boundary_return = (
                old_edges[exit_edge] <= 0.055 and new_edges[exit_edge] <= 0.06
                and outward_speed >= 5 and lateral_offset <= max(5, 0.02 * edge_length)
                and size >= 0.72 and age <= 12.0
            )
            if not plausible_boundary_return:
                return -1.0
        if age > 3.0 and not same_edge and distance > tolerance and appearance_score < 0.93:
            return -1.0

        appearance_weight = self.weights["appearance"] * quality
        size_weight, time_weight = self.weights["size"], self.weights["time"]
        total = appearance_weight + motion_weight + size_weight + edge_weight + time_weight
        if total == 0:
            return -1.0
        time_score = exp(-age / max(self.max_seconds, 1.0))
        return (
            appearance_weight * appearance_score + motion_weight * proximity
            + size_weight * size + edge_weight * edge + time_weight * time_score
        ) / total
