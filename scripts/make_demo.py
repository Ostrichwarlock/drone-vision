"""Generate a reproducible 1080p, 30 FPS synthetic oracle tracking sequence."""

import argparse
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT, FPS, FRAMES = 1920, 1080, 30, 365


def draw_object(frame: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x, y, w, h = box
    cv2.ellipse(frame, (x + w // 2, y + h // 2), (w // 3, h // 3), 0, 0, 360, color, -1)
    cv2.line(frame, (x, y + 3), (x + w - 1, y + h - 4), color, 3)
    cv2.line(frame, (x, y + h - 4), (x + w - 1, y + 3), color, 3)
    for point in ((x + 4, y + 3), (x + w - 5, y + h - 4), (x + 4, y + h - 4), (x + w - 5, y + 3)):
        cv2.circle(frame, point, 3, (42, 43, 47), -1)


def make_demo(directory: Path) -> None:
    truth_path = directory / "gt" / "gt.txt"
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(directory / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not video.isOpened():
        raise RuntimeError("Could not create the synthetic video")
    sky = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for channel, (high, low) in enumerate(((215, 228), (189, 212), (153, 192))):
        sky[:, :, channel] = np.linspace(high, low, HEIGHT, dtype=np.uint8)[:, None]
    title = "SYNTHETIC FIXTURE  |  1920 x 1080  |  30 FPS"
    cv2.putText(sky, title, (35, HEIGHT - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (65, 60, 55), 2, cv2.LINE_AA)
    try:
        with truth_path.open("w", encoding="utf-8") as truth:
            for frame_id in range(1, FRAMES + 1):
                frame = sky.copy()
                objects = [
                    (2, (790 + round(105 * np.sin(frame_id / 25)), 640 + round(43 * np.cos(frame_id / 28)), 63, 25),
                     (55, 81, 149)),
                ]
                if frame_id <= 50:
                    objects.append((1, (1658 + (frame_id - 1) * 4, 366, 57, 26), (44, 51, 52)))
                elif frame_id >= 303:
                    objects.append((1, (1854 - (frame_id - 303) * 4, 382, 57, 26), (44, 51, 52)))
                for identity, box, color in sorted(objects):
                    draw_object(frame, box, color)
                    truth.write(f"{frame_id},{identity},{','.join(map(str, box))},1,1,1\n")
                video.write(frame)
    finally:
        video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New or empty destination for video.mp4 and gt.txt")
    args = parser.parse_args()
    if any((args.directory / filename).exists() for filename in ("video.mp4", "gt/gt.txt")):
        parser.error("Demo video or ground truth already exists in the destination")
    make_demo(args.directory)
