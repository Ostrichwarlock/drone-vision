"""Common detector interface, crop geometry and cross-crop suppression."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from src.types import Box, Detection

Region = tuple[int, int, int, int]


class Detector(Protocol):
    def detect(
        self, frame: np.ndarray, frame_index: int, regions: Sequence[Region] | None = None
    ) -> list[Detection]: ...


@dataclass(frozen=True, slots=True)
class Letterbox:
    region: Region
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int

    def restore(self, box: Box) -> Box | None:
        origin_x, origin_y, limit_x, limit_y = self.region
        x1, y1, x2, y2 = box
        restored = (
            max(origin_x, (x1 - self.pad_x) / self.scale_x + origin_x),
            max(origin_y, (y1 - self.pad_y) / self.scale_y + origin_y),
            min(limit_x, (x2 - self.pad_x) / self.scale_x + origin_x),
            min(limit_y, (y2 - self.pad_y) / self.scale_y + origin_y),
        )
        if restored[2] <= restored[0] or restored[3] <= restored[1]:
            return None
        return restored


def letterboxed_regions(
    frame: np.ndarray, regions: Sequence[Region] | None, target: tuple[int, int]
) -> Iterator[tuple[np.ndarray, Letterbox]]:
    frame_height, frame_width = frame.shape[:2]
    full_frame = (0, 0, frame_width, frame_height)
    for x1, y1, x2, y2 in (full_frame,) if regions is None else regions:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)
        if x1 >= x2 or y1 >= y2:
            continue
        crop = frame[y1:y2, x1:x2]
        target_width, target_height = target
        scale = min(target_width / crop.shape[1], target_height / crop.shape[0])
        width = min(target_width, max(1, round(crop.shape[1] * scale)))
        height = min(target_height, max(1, round(crop.shape[0] * scale)))
        pad_x, pad_y = (target_width - width) // 2, (target_height - height) // 2
        if (width, height) != (crop.shape[1], crop.shape[0]):
            crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        if (width, height) != target:
            crop = cv2.copyMakeBorder(
                crop,
                pad_y,
                target_height - height - pad_y,
                pad_x,
                target_width - width - pad_x,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
        yield crop, Letterbox((x1, y1, x2, y2), width / (x2 - x1), height / (y2 - y1), pad_x, pad_y)


def validate_thresholds(confidence: float, iou_threshold: float, class_id: int, max_detections: int) -> None:
    if not 0 <= confidence <= 1 or not 0 < iou_threshold <= 1:
        raise ValueError("confidence must be in [0, 1] and IoU threshold in (0, 1]")
    if class_id < 0 or max_detections < 1:
        raise ValueError("class_id must be nonnegative and max_detections must be positive")


def suppress(detections: list[Detection], iou_threshold: float, limit: int) -> list[Detection]:
    if len(detections) < 2:
        return detections
    boxes = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in (d.box for d in detections)]
    scores = [d.confidence for d in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou_threshold)
    return [detections[int(index)] for index in np.asarray(indices).ravel()[:limit]]
