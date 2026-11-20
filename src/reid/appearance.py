"""Compact appearance features from a small crop of an observed object."""

from dataclasses import dataclass
from math import ceil, floor

import cv2
import numpy as np

from src.types import Box


@dataclass(slots=True)
class Appearance:
    color: np.ndarray
    gradient: np.ndarray
    quality: float
    crop: np.ndarray


def describe(frame: np.ndarray, box: Box) -> Appearance | None:
    left, top, right, bottom = box
    height, width = frame.shape[:2]
    x1, x2 = max(0, floor(left)), min(width, ceil(right))
    y1, y2 = max(0, floor(top)), min(height, ceil(bottom))
    shortest_side = min(x2 - x1, y2 - y1)
    if shortest_side < 5:
        return None

    crop = cv2.resize(frame[y1:y2, x1:x2], (16, 16), interpolation=cv2.INTER_AREA)
    core = crop[3:-3, 3:-3]
    # Soft histogram bins reduce instability around color boundaries.
    positions = core.reshape(-1, 3).astype(np.float32) / 36.5
    lower = positions.astype(np.int32)
    fraction = positions - lower
    histogram = np.stack([
        np.bincount(lower[:, channel], 1 - fraction[:, channel], minlength=8)
        + np.bincount(np.minimum(lower[:, channel] + 1, 7), fraction[:, channel], minlength=8)
        for channel in range(3)
    ])
    color = np.sqrt(histogram.ravel() / histogram.sum())

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = gray[1:-1, 2:] - gray[1:-1, :-2]
    dy = gray[2:, 1:-1] - gray[:-2, 1:-1]
    orientations = ((np.arctan2(dy, dx) % np.pi) * (8 / np.pi)).astype(np.int32) % 8
    gradient = np.bincount(orientations.ravel(), np.hypot(dx, dy).ravel(), minlength=8)
    gradient = np.sqrt(gradient / gradient.sum()) if gradient.any() else gradient
    color_contrast = float(np.ptp(core.mean(axis=(0, 1))))
    detail = min(1.0, 0.4 + max(float(gray.std()), color_contrast / 3) / 18)
    quality = min(shortest_side / 18, 1.0) * detail
    return Appearance(color, gradient, quality, crop)


def similarity(first: Appearance, second: Appearance) -> float:
    color = float(first.color @ second.color)
    if first.gradient.any() and second.gradient.any():
        return 0.8 * color + 0.2 * float(first.gradient @ second.gradient)
    return color
