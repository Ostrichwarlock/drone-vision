"""OpenCV DNN detector for raw Ultralytics YOLOv8 and YOLO11 exports."""

from pathlib import Path
from time import perf_counter
from typing import Sequence

import cv2
import numpy as np

from src.detectors.base import Region, letterboxed_regions, suppress, validate_thresholds
from src.types import Detection


class OnnxDetector:
    mode = "onnx"

    def __init__(
        self,
        path: str | Path,
        *,
        input_size: int = 640,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        class_id: int = 0,
        max_detections: int = 100,
    ):
        validate_thresholds(confidence, iou_threshold, class_id, max_detections)
        if input_size < 32 or input_size % 32:
            raise ValueError("ONNX input_size must be a positive multiple of 32")
        path = Path(path)
        if path.suffix.lower() != ".onnx" or not path.is_file():
            raise ValueError(f"Expected an existing .onnx model file, got {path}")
        try:
            self._net = cv2.dnn.readNetFromONNX(str(path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as exc:
            raise ValueError(f"OpenCV could not load {path}. Use the documented raw YOLO ONNX export.") from exc
        self.input_size = input_size
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.class_id = class_id
        self.max_detections = max_detections
        self.last_inference_seconds = 0.0

    def detect(
        self, frame: np.ndarray, frame_index: int, regions: Sequence[Region] | None = None
    ) -> list[Detection]:
        self.last_inference_seconds = 0.0
        target_size = (self.input_size, self.input_size)
        detections = []
        for image, geometry in letterboxed_regions(frame, regions, target_size):
            blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, swapRB=True)
            self._net.setInput(blob)
            start = perf_counter()
            try:
                output = self._net.forward()
            except cv2.error as exc:
                raise RuntimeError(
                    f"ONNX forward failed at {self.input_size}px. Match the export's input size."
                ) from exc
            self.last_inference_seconds += perf_counter() - start
            for cx, cy, width, height, score in self._candidates(output):
                if not np.isfinite((cx, cy, width, height)).all() or width <= 0 or height <= 0:
                    continue
                box = geometry.restore(
                    (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
                )
                if box is not None:
                    detections.append(Detection(tuple(map(float, box)), float(score), self.class_id))
        return suppress(detections, self.iou_threshold, self.max_detections)

    def _candidates(self, output: np.ndarray) -> np.ndarray:
        if output.ndim != 3 or output.shape[0] != 1 or not 5 <= output.shape[1] < output.shape[2]:
            raise ValueError(
                f"Unsupported ONNX output shape {output.shape}. "
                "Expected a raw [1, 4 + class_count, anchors] YOLOv8 or YOLO11 export."
            )
        classes = output[0, 4:]
        if self.class_id >= len(classes):
            raise ValueError(f"class_id {self.class_id} is outside the model's {len(classes)} classes")
        scores = classes[self.class_id]
        candidates = np.flatnonzero((scores > 0) & (scores >= self.confidence) & (scores <= 1))
        if len(classes) > 1 and len(candidates):
            candidates = candidates[np.argmax(classes[:, candidates], axis=0) == self.class_id]
        return np.column_stack((output[0, :4, candidates], scores[candidates]))
