"""Competition text and annotated video writers."""

from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

from src.types import Box


def clip_box(box: Box, width: int, height: int) -> Box | None:
    left, top, right, bottom = box
    left, right = max(0.0, left), min(float(width), right)
    top, bottom = max(0.0, top), min(float(height), bottom)
    if right - left < 0.001 or bottom - top < 0.001:
        return None
    return left, top, right, bottom


def write_rows(
    output: TextIO,
    frame: int,
    items: Iterable[tuple[int, Box, float]],
    image_size: tuple[int, int],
) -> None:
    for identity, box, confidence in items:
        clipped = clip_box(box, *image_size)
        if clipped is None:
            continue
        left, top, right, bottom = clipped
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid confidence {confidence!r} on frame {frame}")
        output.write(
            f"{frame},{identity},{left:.3f},{top:.3f},"
            f"{right-left:.3f},{bottom-top:.3f},{confidence:.6f}\n"
        )


class AnnotatedVideo:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None

    def write(self, image: np.ndarray) -> None:
        size = (image.shape[1], image.shape[0])
        if self.writer is None:
            self.size = size
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {self.path}")
        elif size != self.size:
            raise RuntimeError("Source resolution changed while recording annotated video")
        self.writer.write(image)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
