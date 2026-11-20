"""Draw stable track labels and short trajectories on video frames."""

from collections import deque

import cv2
import numpy as np

from src.pipeline.output import clip_box
from src.types import Track

PALETTE = (
    (85, 237, 255), (247, 180, 86), (143, 241, 151), (207, 138, 250),
    (211, 238, 98), (117, 159, 255), (233, 159, 171), (120, 232, 205),
)


class Overlay:
    def __init__(self) -> None:
        self.paths: dict[int, deque[tuple[int, int]]] = {}
        self.reidentified_at: dict[int, float] = {}

    def draw(
        self, image: np.ndarray, tracks: list[Track], timestamp: float, fps: float, mode: str
    ) -> np.ndarray:
        height, width = image.shape[:2]
        scale = max(0.45, min(0.8, width / 2400))
        self.paths = {track.id: self.paths.get(track.id, deque(maxlen=24)) for track in tracks}
        self.reidentified_at = {
            identity: at for identity, at in self.reidentified_at.items() if timestamp - at < 1.0
        }
        for track in tracks:
            box = clip_box(track.box, width, height)
            if box is None:
                continue
            left, top, right, bottom = map(round, box)
            color = PALETTE[(track.id - 1) % len(PALETTE)]
            if track.reidentified:
                self.reidentified_at[track.id] = timestamp
            trajectory = self.paths[track.id]
            trajectory.append(((left + right) // 2, (top + bottom) // 2))
            if len(trajectory) > 1:
                cv2.polylines(image, [np.asarray(trajectory, dtype=np.int32)], False, color, 1)
            cv2.rectangle(image, (left, top), (right - 1, bottom - 1), color, 2)
            label = f"Drone {track.id}  {track.confidence:.2f}"
            if track.id in self.reidentified_at:
                label += "  Re-identified"
            _label(image, label, left, max(18, top - 5), scale, color)
        prefix = "Oracle diagnostic | " if mode == "oracle" else ""
        _label(image, f"{prefix}{fps:.1f} processing FPS", 8, 23, scale, (240, 240, 240))
        return image


def _label(
    image: np.ndarray, label: str, x: int, baseline: int, scale: float, color: tuple[int, int, int]
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), _ = cv2.getTextSize(label, font, scale, 1)
    x = max(0, min(x, image.shape[1] - width - 8))
    cv2.rectangle(image, (x, max(0, baseline - height - 5)), (x + width + 7, baseline + 4), (18, 20, 24), -1)
    cv2.putText(image, label, (x + 3, baseline), font, scale, color, 1, cv2.LINE_AA)
