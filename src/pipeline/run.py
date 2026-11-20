"""Connect capture, detection, tracking and output without future frames."""

from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
import json
import logging
import math
import resource
import sys
from tempfile import NamedTemporaryFile
from time import monotonic, process_time
from typing import Protocol, cast

import cv2
import numpy as np

from src.detectors.base import Detector, Region, suppress
from src.detectors.onnx import OnnxDetector
from src.pipeline.capture import LiveFrames, file_frames
from src.pipeline.output import AnnotatedVideo, write_rows
from src.pipeline.timing import Timings
from src.types import Detection, Track
from src.visualization import Overlay


class Tracker(Protocol):
    def update(
        self, detections: list[Detection] | None, frame: np.ndarray,
        timestamp: float, frame_index: int,
    ) -> list[Track]: ...


@dataclass(frozen=True, slots=True)
class Settings:
    source: str | int
    source_type: str
    mode: str
    interval: int = 1
    tiles: int = 1
    tile_overlap: float = 0.20
    max_frames: int | None = None
    expected_frames: int | None = None
    fallback_live_fps: float = 30
    predictions: Path | None = None
    detections: Path | None = None
    annotated: Path | None = None
    stats: Path | None = None
    display: bool = False
    tile_fallback_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.tile_fallback_confidence is None:
            return
        if not 0 < self.tile_fallback_confidence <= 1:
            raise ValueError("Tile fallback confidence must be in (0, 1]")
        if self.mode != "onnx":
            raise ValueError("Tile fallback requires the ONNX backend")
        if self.tiles != 1:
            raise ValueError("Tile fallback starts with the full frame and requires --tiles 1")


def grid_regions(
    width: int, height: int, tiles: int, overlap: float
) -> list[Region] | None:
    if tiles == 1:
        return None
    pad_x = round(width * overlap / (2 * tiles))
    pad_y = round(height * overlap / (2 * tiles))
    return [
        (
            max(0, col * width // tiles - pad_x),
            max(0, row * height // tiles - pad_y),
            min(width, (col + 1) * width // tiles + pad_x),
            min(height, (row + 1) * height // tiles + pad_y),
        )
        for row in range(tiles) for col in range(tiles)
    ]


def run(settings: Settings, detector: Detector, tracker: Tracker) -> dict:
    """Publish outputs only after the run and frame-count checks finish."""
    with ExitStack() as cleanup:
        temporary = {}
        for name in ("predictions", "detections", "annotated", "stats"):
            path = getattr(settings, name)
            if path is None:
                continue
            with NamedTemporaryFile(dir=path.parent, prefix=".drone-", suffix=path.suffix, delete=False) as file:
                temporary[name] = Path(file.name)
            cleanup.callback(temporary[name].unlink, missing_ok=True)
        result = _process(replace(settings, **temporary), detector, tracker)
        for name, path in temporary.items():
            path.replace(getattr(settings, name))
        return result


def _process(settings: Settings, detector: Detector, tracker: Tracker) -> dict:
    log = logging.getLogger(__name__)
    live = settings.source_type != "video"
    if settings.display and cv2.currentUIFramework() == "":
        raise RuntimeError("Live display requires a GUI-enabled OpenCV installation")

    if settings.source_type == "stream":
        capture = cv2.VideoCapture(
            settings.source, cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 4000],
        )
    else:
        capture = cv2.VideoCapture(settings.source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open {settings.source_type} input")

    timings = Timings()
    count = last_index = dropped = calls = track_rows = recoveries = fallback_frames = 0
    fallback_confidence = settings.tile_fallback_confidence
    last_capture_time = 0.0
    size = (0, 0)
    started = monotonic()
    cpu_started = process_time()
    with ExitStack() as resources:
        if not live:
            resources.callback(capture.release)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            if not live:
                raise RuntimeError("Source video FPS is unavailable")
            fps = settings.fallback_live_fps
            log.warning("Capture FPS unavailable, using %.3f for recording", fps)
        declared = capture.get(cv2.CAP_PROP_FRAME_COUNT) if not live else 0
        if live:
            # The reader owns capture until it exits its read call.
            frames = iter(resources.enter_context(LiveFrames(capture)))
        else:
            frames = file_frames(capture, fps)
        predictions = resources.enter_context(settings.predictions.open("w", encoding="utf-8")) if settings.predictions else None
        raw = resources.enter_context(settings.detections.open("w", encoding="utf-8")) if settings.detections else None
        video = AnnotatedVideo(settings.annotated, fps) if settings.annotated else None
        if video:
            resources.callback(video.close)
        overlay = Overlay() if video or settings.display else None
        if settings.display:
            resources.callback(cv2.destroyAllWindows)
        next_detection = 1
        while settings.max_frames is None or count < settings.max_frames:
            try:
                frame = next(frames)
            except StopIteration:
                break
            except KeyboardInterrupt:
                if not live:
                    raise
                break
            frame_started = monotonic()
            timings.record("capture_read", frame.read_seconds)
            if live:
                timings.record("capture_to_process", frame_started - frame.captured_at)
                dropped += frame.index - last_index - 1
                last_capture_time = frame.timestamp
            last_index = frame.index
            count += 1
            size = frame.image.shape[1], frame.image.shape[0]
            detections = None
            if frame.index >= next_detection:
                regions = grid_regions(*size, settings.tiles, settings.tile_overlap)
                start = monotonic()
                detections = detector.detect(frame.image, frame.index, regions=regions)
                inference_time = getattr(detector, "last_inference_seconds", None)
                if fallback_confidence is not None and not any(
                    detection.confidence >= fallback_confidence for detection in detections
                ):
                    onnx = cast(OnnxDetector, detector)
                    tiles = grid_regions(*size, 2, settings.tile_overlap)
                    tiled = onnx.detect(frame.image, frame.index, regions=tiles)
                    inference_time = (inference_time or 0.0) + onnx.last_inference_seconds
                    detections = suppress(detections + tiled, onnx.iou_threshold, onnx.max_detections)
                    fallback_frames += 1
                timings.record("detector", monotonic() - start)
                if settings.mode != "oracle" and inference_time is not None:
                    timings.record("model_inference", inference_time)
                calls += 1
                next_detection = frame.index + settings.interval
                if raw is not None:
                    write_rows(raw, frame.index, ((i, d.box, d.confidence) for i, d in enumerate(detections, 1)), size)
            start = monotonic()
            tracks = tracker.update(detections, frame.image, frame.timestamp, frame.index)
            tracker_elapsed = monotonic() - start
            timings.record("tracker_including_reid", tracker_elapsed)
            reid_time = getattr(tracker, "last_reid_seconds", None)
            if reid_time is not None:
                timings.record("reid", reid_time)
                timings.record("tracker_excluding_reid", max(0.0, tracker_elapsed - reid_time))
            track_rows += len(tracks)
            recoveries += sum(track.reidentified for track in tracks)
            if predictions is not None:
                write_rows(predictions, frame.index, ((t.id, t.box, t.confidence) for t in tracks), size)
            stop_preview = False
            if overlay:
                start = monotonic()
                image = overlay.draw(frame.image, tracks, frame.timestamp, count / (start - started), settings.mode)
                timings.record("overlay", monotonic() - start)
                if video:
                    start = monotonic()
                    video.write(image)
                    timings.record("video_write", monotonic() - start)
                if settings.display:
                    cv2.imshow("Drone Vision", image)
                    stop_preview = cv2.waitKey(1) & 255 in (ord("q"), 27)
            timings.record("frame_processing", monotonic() - frame_started)
            if live:
                timings.record("capture_to_output", monotonic() - frame.captured_at)
            if stop_preview:
                break

    if count == 0:
        raise RuntimeError("Input did not yield a video frame")
    if settings.expected_frames is not None and count != settings.expected_frames:
        raise RuntimeError(f"Decoded {count} frames, expected {settings.expected_frames}")
    if (not live and not settings.max_frames and not settings.display
            and math.isfinite(declared) and declared > 0 and count != round(declared)):
        raise RuntimeError(f"Decoded {count} frames, but the source reports {round(declared)}")
    elapsed = monotonic() - started
    cpu_elapsed = process_time() - cpu_started
    timings_ms = timings.summary()
    memory_divisor = (1024 * 1024) if sys.platform == "darwin" else 1024
    result = {
        "detector_mode": settings.mode,
        "source_type": settings.source_type,
        "source_fps": fps,
        "live_observed_capture_fps": (last_index - 1) / last_capture_time if last_capture_time else None,
        "frames": count,
        "last_source_frame": last_index,
        "frames_dropped_live": dropped,
        "source_size": list(size),
        "detector_input_size": getattr(detector, "input_size", None),
        "detector_interval": settings.interval,
        "tiles_per_axis": settings.tiles,
        "tile_fallback_confidence": fallback_confidence,
        "tile_fallback_frames": fallback_frames,
        "detector_calls": calls,
        "tracks_emitted": track_rows,
        "reidentified_events": recoveries,
        "elapsed_seconds": elapsed,
        "processing_fps": count / elapsed,
        "process_cpu_percent": 100 * cpu_elapsed / elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / memory_divisor,
        "timings_ms": timings_ms,
    }
    if settings.stats:
        settings.stats.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
