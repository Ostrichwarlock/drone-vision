"""Shared detector and tracker records in original image coordinates."""

from dataclasses import dataclass

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    confidence: float
    class_id: int = 0


@dataclass(frozen=True, slots=True)
class Track:
    id: int
    box: Box
    confidence: float
    reidentified: bool = False
