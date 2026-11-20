from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

from scripts.run_video import build_parser, settings_for
from src.pipeline.run import Settings, grid_regions, run
from src.types import Detection, Track


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (96, 64))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
    writer.release()


class StubDetector:
    def detect(self, frame, frame_index, regions=None):
        assert frame.shape == (64, 96, 3)
        return [Detection((-1, 4, 8, 13), 0.90)] if frame_index != 3 else []


class StubTracker:
    def __init__(self):
        self.seen = []

    def update(self, detections, frame, timestamp, frame_index):
        self.seen.append((frame_index, timestamp, detections))
        return [Track(4, (-1, 4, 8, 13), 0.90)] if detections else []


def test_offline_output_preserves_frame_indices_and_video_fps(tmp_path):
    source = tmp_path / "source.mp4"
    make_video(source)
    pred = tmp_path / "pred.txt"
    annotated = tmp_path / "annotated.mp4"
    tracker = StubTracker()
    settings = Settings(str(source), "video", "onnx", interval=2, expected_frames=5,
                        predictions=pred, annotated=annotated)
    stats = run(settings, StubDetector(), tracker)
    assert stats["frames"] == 5
    assert stats["source_fps"] == pytest.approx(30)
    assert stats["detector_calls"] == 3
    assert stats["frames_dropped_live"] == 0
    assert [entry[0] for entry in tracker.seen] == [1, 2, 3, 4, 5]
    assert tracker.seen[1][1] == pytest.approx(1 / 30)
    assert tracker.seen[1][2] is None
    assert tracker.seen[2][2] == []
    assert pred.read_text().splitlines() == [
        "1,4,0.000,4.000,8.000,9.000,0.900000",
        "5,4,0.000,4.000,8.000,9.000,0.900000",
    ]
    capture = cv2.VideoCapture(str(annotated))
    try:
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 5
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30)
    finally:
        capture.release()


def test_wrong_frame_count_does_not_publish_partial_output(tmp_path):
    source = tmp_path / "source.mp4"
    make_video(source)
    pred = tmp_path / "pred.txt"
    pred.write_text("previous output\n")
    settings = Settings(str(source), "video", "onnx", expected_frames=6, predictions=pred)
    with pytest.raises(RuntimeError, match="Decoded 5 frames, expected 6"):
        run(settings, StubDetector(), StubTracker())
    assert pred.read_text() == "previous output\n"
    assert not list(tmp_path.glob(".drone-*"))


def test_grid_regions_cover_frame_with_overlap():
    assert grid_regions(1920, 1080, 1, 0.2) is None
    regions = grid_regions(1920, 1080, 2, 0.2)
    assert regions == [(0, 0, 1056, 594), (864, 0, 1920, 594),
                       (0, 486, 1056, 1080), (864, 486, 1920, 1080)]


def test_tile_fallback_merges_detections_and_counts_frames_and_inference(tmp_path):
    source = tmp_path / "source.mp4"
    make_video(source)
    raw = tmp_path / "raw.txt"
    tracker = StubTracker()
    shared_box = (10, 10, 20, 20)
    full_only = Detection((40, 40, 48, 48), 0.23)
    tile_only = Detection((60, 20, 68, 28), 0.61)
    stronger = Detection(shared_box, 0.81)

    class Detector:
        iou_threshold = 0.45
        max_detections = 3

        def __init__(self):
            self.calls = []

        def detect(self, frame, frame_index, regions=None):
            self.calls.append((frame_index, regions))
            self.last_inference_seconds = 0.001 if regions is None else 0.004
            if regions is not None:
                return [stronger, tile_only, Detection((70, 40, 78, 48), 0.21)]
            if frame_index == 3:
                return [Detection(shared_box, 0.20), full_only]
            return [Detection(shared_box, 0.25)]

    detector = Detector()
    stats = run(Settings(str(source), "video", "onnx", interval=2, detections=raw,
                         tile_fallback_confidence=0.25), detector, tracker)
    assert detector.calls == [(1, None), (3, None), (3, grid_regions(96, 64, 2, 0.20)), (5, None)]
    assert tracker.seen[2][2] == [stronger, tile_only, full_only]
    assert stats["detector_calls"] == 3
    assert stats["tile_fallback_frames"] == 1
    assert stats["tile_fallback_confidence"] == 0.25
    assert stats["timings_ms"]["model_inference"]["count"] == 3
    assert stats["timings_ms"]["model_inference"]["mean"] == pytest.approx(7 / 3)
    assert [row.split(",")[-1] for row in raw.read_text().splitlines() if row.startswith("3,")] == [
        "0.810000", "0.610000", "0.230000",
    ]


def test_tile_fallback_cli_requires_onnx_and_full_frame():
    parser = build_parser()
    args = parser.parse_args(["--camera", "0", "--tile-fallback-confidence", "0.25"])
    assert settings_for(args, {}, "onnx").tile_fallback_confidence == 0.25
    with pytest.raises(ValueError, match="requires the ONNX backend"):
        settings_for(args, {}, "hailo")
    args = parser.parse_args(["--camera", "0", "--tiles", "2", "--tile-fallback-confidence", "0.25"])
    with pytest.raises(ValueError, match="requires --tiles 1"):
        settings_for(args, {}, "onnx")


def test_live_backpressure_keeps_latest_frame_and_original_index(monkeypatch):
    producer_waiting = Event()
    finish = Event()
    image = np.zeros((64, 96, 3), dtype=np.uint8)

    class Camera:
        frames_read = 0

        def isOpened(self):
            return True

        def get(self, _):
            return 30

        def read(self):
            self.frames_read += 1
            if self.frames_read <= 3:
                return True, image
            producer_waiting.set()
            assert finish.wait(timeout=5)
            return False, None

        def release(self):
            pass

    class DelayedDetector(StubDetector):
        def detect(self, frame, frame_index, regions=None):
            assert producer_waiting.wait(timeout=5)
            finish.set()
            return super().detect(frame, frame_index, regions)

    # Delay the consumer before its first read to reproduce a full buffer.
    from src.pipeline.capture import LiveFrames

    enter = LiveFrames.__enter__

    def start_with_full_buffer(self):
        result = enter(self)
        assert producer_waiting.wait(timeout=5)
        return result

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: Camera())
    monkeypatch.setattr(LiveFrames, "__enter__", start_with_full_buffer)
    tracker = StubTracker()
    stats = run(Settings(0, "camera", "onnx", max_frames=1), DelayedDetector(), tracker)
    assert stats["frames"] == 1
    assert stats["last_source_frame"] == 3
    assert stats["frames_dropped_live"] == 2
    assert tracker.seen[0][0] == 3
