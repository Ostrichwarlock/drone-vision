"""Tracker development backend replaying known boxes on the current frame."""

from pathlib import Path
from typing import Sequence

import numpy as np

from src.detectors.base import Region
from src.mot import read_mot
from src.types import Detection


class AnnotatedDetector:
    mode = "annotation_replay"
    input_size = None
    last_inference_seconds = 0.0

    def __init__(self, path: str | Path, *, coordinate_offset: float = 0.0):
        rows = read_mot(path, ground_truth=True, coordinate_offset=coordinate_offset)
        self._detections = {
            frame: tuple(
                sorted(
                    (Detection(row.box, row.confidence) for row in frame_rows),
                    key=lambda detection: (detection.box, detection.confidence),
                )
            )
            for frame, frame_rows in rows.items()
        }

    def detect(
        self, frame: np.ndarray, frame_index: int, regions: Sequence[Region] | None = None
    ) -> list[Detection]:
        if regions is not None:
            raise ValueError("Annotation replay requires full-frame detections")
        return list(self._detections.get(frame_index, ()))
