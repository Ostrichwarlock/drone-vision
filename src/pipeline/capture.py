"""Sequential file input and latest-frame live input."""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count
from threading import Condition, Thread
from time import monotonic

import cv2
import numpy as np


@dataclass(slots=True)
class Frame:
    index: int
    timestamp: float
    image: np.ndarray
    read_seconds: float
    captured_at: float


def file_frames(capture: cv2.VideoCapture, fps: float) -> Iterator[Frame]:
    for index in count(1):
        started = monotonic()
        ok, image = capture.read()
        ended = monotonic()
        if not ok:
            return
        yield Frame(index, (index - 1) / fps, image, ended - started, ended)


class LiveFrames:
    """A one-frame buffer. A new capture replaces any unconsumed frame."""

    def __init__(self, capture: cv2.VideoCapture):
        self.capture = capture
        self.condition = Condition()
        self.latest: Frame | None = None
        self.finished = False
        self.stopping = False
        self.error: Exception | None = None
        self.reader = Thread(target=self._read, name="video-capture", daemon=True)

    def __enter__(self) -> "LiveFrames":
        self.reader.start()
        return self

    def __exit__(self, *_: object) -> None:
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.reader.join(timeout=1.0)

    def __iter__(self) -> Iterator[Frame]:
        while True:
            with self.condition:
                ready = self.condition.wait_for(
                    lambda: self.latest is not None or self.finished, timeout=15.0
                )
                if not ready:
                    raise RuntimeError("Live capture returned no frames for 15 seconds")
                if self.latest is None:
                    if self.error:
                        raise RuntimeError("Live capture failed") from self.error
                    return
                frame = self.latest
                self.latest = None
            yield frame

    def _read(self) -> None:
        first_capture = None
        try:
            for index in count(1):
                with self.condition:
                    if self.stopping:
                        return
                started = monotonic()
                ok, image = self.capture.read()
                ended = monotonic()
                if not ok:
                    raise RuntimeError("Camera or stream stopped delivering frames")
                if first_capture is None:
                    first_capture = ended
                with self.condition:
                    self.latest = Frame(index, ended - first_capture, image, ended - started, ended)
                    self.condition.notify_all()
        except Exception as exc:
            self.error = exc
        finally:
            self.capture.release()
            with self.condition:
                self.finished = True
                self.condition.notify_all()
