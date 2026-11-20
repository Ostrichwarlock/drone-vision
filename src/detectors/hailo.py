"""Native HailoRT inference for RGB YOLO HEFs with per-class NMS."""

from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Sequence

import cv2
import numpy as np

from src.detectors.base import Region, letterboxed_regions, suppress, validate_thresholds
from src.types import Detection


def _architecture(value) -> str:
    return str(getattr(value, "name", value)).lower().replace("-", "")


class HailoDetector:
    mode = "hailo"

    def __init__(
        self,
        path: str | Path,
        *,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        class_id: int = 0,
        max_detections: int = 100,
        expected_arch: str | None = None,
    ):
        validate_thresholds(confidence, iou_threshold, class_id, max_detections)
        path = Path(path)
        if path.suffix.lower() != ".hef" or not path.is_file():
            raise ValueError(f"Expected an existing .hef model file, got {path}")
        if expected_arch is not None and _architecture(expected_arch) not in {"hailo8", "hailo8l", "hailo10h"}:
            raise ValueError("expected_arch must be hailo8, hailo8l or hailo10h")
        try:
            hailo = import_module("hailo_platform")
        except (ImportError, OSError) as exc:
            raise ImportError("Hailo needs the device's HailoRT libraries and matching hailo_platform Python bindings") from exc

        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.class_id = class_id
        self.max_detections = max_detections
        self.last_inference_seconds = 0.0
        self.architecture = None
        self._error = hailo.HailoRTException

        with ExitStack() as stack:
            try:
                hef = hailo.HEF(str(path))
                inputs, outputs = hef.get_input_vstream_infos(), hef.get_output_vstream_infos()
                if len(hef.get_network_group_names()) != 1 or len(inputs) != 1 or len(outputs) != 1:
                    raise ValueError("HEF must have one network group, one image input and one detector output")
                image_info, detection_info = inputs[0], outputs[0]
                shape = tuple(image_info.shape)
                if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0 or image_info.format.type != hailo.FormatType.UINT8:
                    raise ValueError("HEF must accept a three-channel UINT8 HWC image")
                nms_orders = tuple(
                    order
                    for name in ("HAILO_NMS_BY_CLASS", "HAILO_NMS")
                    if (order := getattr(hailo.FormatOrder, name, None)) is not None
                )
                if detection_info.format.order not in nms_orders:
                    raise ValueError("HEF must have a per-class output with integrated detection NMS")
                if self.class_id >= detection_info.nms_shape.number_of_classes:
                    raise ValueError(f"class_id {self.class_id} is outside the HEF's class range")
                self.input_size = (shape[1], shape[0])

                params = hailo.VDevice.create_params()
                params.scheduling_algorithm = hailo.HailoSchedulingAlgorithm.ROUND_ROBIN
                device = stack.enter_context(hailo.VDevice(params))
                if expected_arch is not None:
                    with ExitStack() as physical_devices:
                        architectures = {
                            _architecture(physical_devices.enter_context(chip).control.identify().device_architecture)
                            for chip in device.get_physical_devices()
                        }
                    if architectures != {_architecture(expected_arch)}:
                        found = ", ".join(sorted(architectures)) or "no devices"
                        raise ValueError(f"Expected {expected_arch}, detected {found}")
                    self.architecture = architectures.pop()

                model = device.create_infer_model(str(path))
                model.set_batch_size(1)
                model_input = model.input()
                model_input.set_format_type(hailo.FormatType.UINT8)
                model_input.set_format_order(hailo.FormatOrder.NHWC)
                output = model.output()
                output.set_format_type(hailo.FormatType.FLOAT32)
                self._configured = stack.enter_context(model.configure())
                self._bindings = self._configured.create_bindings(
                    output_buffers={detection_info.name: np.empty(output.shape, dtype=np.float32)}
                )
            except self._error as exc:
                raise RuntimeError(
                    f"HailoRT could not configure {path}. Check the device and HEF architecture with hailortcli: {exc}"
                ) from exc
            self._resources = stack.pop_all()

    def detect(
        self, frame: np.ndarray, frame_index: int, regions: Sequence[Region] | None = None
    ) -> list[Detection]:
        if self._bindings is None:
            raise RuntimeError("The Hailo detector is closed")
        self.last_inference_seconds = 0.0
        width, height = self.input_size
        detections = []
        for image, geometry in letterboxed_regions(frame, regions, self.input_size):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                self._bindings.input().set_buffer(rgb)
                start = perf_counter()
                self._configured.run([self._bindings], 10_000)
                self.last_inference_seconds += perf_counter() - start
                per_class = self._bindings.output().get_buffer()
            except self._error as exc:
                raise RuntimeError(f"HailoRT inference failed: {exc}") from exc
            if not isinstance(per_class, (list, tuple)) or len(per_class) <= self.class_id:
                raise RuntimeError("HailoRT returned an unexpected per-class NMS result")
            rows = np.asarray(per_class[self.class_id])
            if rows.size == 0:
                continue
            if rows.ndim != 2 or rows.shape[1] != 5:
                raise RuntimeError("HailoRT NMS expected rows of [ymin, xmin, ymax, xmax, score]")
            for y1, x1, y2, x2, confidence in rows:
                if not 0 < confidence <= 1 or confidence < self.confidence:
                    continue
                if not np.isfinite((x1, y1, x2, y2)).all():
                    continue
                box = geometry.restore((x1 * width, y1 * height, x2 * width, y2 * height))
                if box is not None:
                    detections.append(Detection(tuple(map(float, box)), float(confidence), self.class_id))
        return suppress(detections, self.iou_threshold, self.max_detections)

    def close(self) -> None:
        self._bindings = None
        self._resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
