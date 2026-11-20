"""Inspect local practice footage or validate a YOLO bounding-box dataset."""

import argparse
import csv
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    minimum, median, maximum = np.percentile(values, [0, 50, 100])
    return {"min": float(minimum), "median": float(median), "max": float(maximum)}


def inspect_practice(root: Path, decode: bool) -> dict:
    videos = {}
    for video_path in sorted(root.rglob("video.mp4")):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        try:
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or not math.isfinite(fps) or width < 1 or height < 1:
                raise ValueError(f"Invalid video metadata for {video_path}")
            decoded = None
            if decode:
                decoded = 0
                while capture.grab():
                    decoded += 1
        finally:
            capture.release()
        annotation_path = video_path.parent / "gt" / "gt.txt"
        intervals = defaultdict(set)
        widths, heights = [], []
        clipped_boxes = 0
        if annotation_path.is_file():
            with annotation_path.open(encoding="utf-8-sig", newline="") as handle:
                for number, row in enumerate(csv.reader(handle), 1):
                    if len(row) != 9 or [item.strip() for item in row[6:]] != ["1", "1", "1"]:
                        raise ValueError(f"{annotation_path}:{number} expected nine fields ending in 1,1,1")
                    frame_id, identity = map(int, row[:2])
                    x, y, w, h = map(float, row[2:6])
                    if (not 1 <= frame_id <= (decoded or frames) or identity <= 0
                            or not all(map(math.isfinite, (x, y, w, h))) or min(w, h) <= 0
                            or frame_id in intervals[identity]):
                        raise ValueError(f"{annotation_path}:{number} invalid or duplicate box")
                    intervals[identity].add(frame_id)
                    widths.append(w)
                    heights.append(h)
                    clipped_boxes += x < 0 or y < 0 or x + w > width or y + h > height
        longest_gap = 0
        for sequence in intervals.values():
            ordered = sorted(sequence)
            longest_gap = max(longest_gap, max((b - a - 1 for a, b in zip(ordered, ordered[1:])), default=0))
        videos[str(video_path.relative_to(root))] = {
            "resolution": [width, height], "fps": fps, "declared_frames": frames, "decoded_frames": decoded,
            "frame_count_matches": None if decoded is None else decoded == frames,
            "annotations": len(widths) if annotation_path.is_file() else None,
            "identities": len(intervals), "label_file_present": (video_path.parent / "gt" / "labels.txt").is_file(),
            "box_width_pixels": distribution(widths), "box_height_pixels": distribution(heights),
            "boxes_outside_coordinate_bound": clipped_boxes,
            "longest_gap_frames": longest_gap, "longest_gap_seconds": longest_gap / fps,
        }
    if not videos:
        raise ValueError(f"No practice video.mp4 files found under {root}")
    return {"videos": videos}


def label_for(image: Path, root: Path) -> tuple[Path, str]:
    relative = image.relative_to(root)
    if "images" not in relative.parts[:-1]:
        raise ValueError(f"Image must be in a YOLO images directory: {image}")
    parts = list(relative.parts)
    index = parts.index("images")
    split = parts[index - 1] if index else (parts[1] if len(parts) >= 3 else "unsplit")
    parts[index] = "labels"
    return root.joinpath(*parts).with_suffix(".txt"), split


def inspect_yolo(root: Path, hash_images: bool) -> dict:
    names = {}
    for config in (root / "data.yaml", root / "dataset.yaml"):
        if not config.is_file():
            continue
        metadata = yaml.safe_load(config.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid YOLO configuration: {config}")
        raw_names = metadata.get("names", {})
        if isinstance(raw_names, list):
            names = dict(enumerate(raw_names))
        elif isinstance(raw_names, dict):
            names = {int(key): label for key, label in raw_names.items()}
        else:
            raise ValueError(f"Expected a list or mapping of class names: {config}")
        break
    images = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and "images" in path.relative_to(root).parts[:-1]
    ]
    if not images:
        raise ValueError(f"No YOLO images directories with images found under {root}")
    split_counts = defaultdict(Counter)
    classes = Counter()
    errors = []
    widths, heights = [], []
    resolutions = Counter()
    used_labels = set()
    content_splits = defaultdict(set)
    for image in images:
        label, split = label_for(image, root)
        counts = split_counts[split]
        counts["images"] += 1
        frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if frame is None:
            errors.append(f"Cannot decode {image.relative_to(root)}")
            continue
        height, width = frame.shape[:2]
        resolutions[f"{width}x{height}"] += 1
        if hash_images:
            digest = sha256(image.read_bytes()).hexdigest()
            content_splits[digest].add(split)
        if not label.is_file():
            counts["images_without_label_file"] += 1
            continue
        used_labels.add(label)
        lines = [line for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not lines:
            counts["images_with_empty_labels"] += 1
        for number, line in enumerate(lines, 1):
            try:
                category, *values = line.split()
                category = int(category)
                x, y, w, h = map(float, values)
                if (category < 0 or (names and category not in names) or not all(map(math.isfinite, (x, y, w, h)))
                        or not 0 < w <= 1 or not 0 < h <= 1
                        or not w / 2 - 1e-6 <= x <= 1 - w / 2 + 1e-6
                        or not h / 2 - 1e-6 <= y <= 1 - h / 2 + 1e-6):
                    raise ValueError
            except ValueError:
                errors.append(f"{label.relative_to(root)}:{number} invalid normalized YOLO box")
                continue
            classes[category] += 1
            counts["annotations"] += 1
            widths.append(w * width)
            heights.append(h * height)
    orphan_labels = sum(
        path not in used_labels for path in root.rglob("*.txt")
        if "labels" in path.relative_to(root).parts[:-1]
    )
    return {
        "splits": dict(split_counts), "class_names": names, "class_annotation_counts": dict(classes),
        "image_resolution_counts": dict(resolutions),
        "box_width_pixels": distribution(widths), "box_height_pixels": distribution(heights),
        "orphan_labels": orphan_labels, "error_count": len(errors), "first_errors": errors[:30],
        "duplicate_images_across_splits": (
            sum(len(splits) > 1 for splits in content_splits.values()) if hash_images else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("practice", "yolo"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--decode", action="store_true", help="Count actual frames in each practice video")
    parser.add_argument("--hash-images", action="store_true", help="Check exact image duplicates across YOLO splits")
    args = parser.parse_args()
    try:
        result = inspect_practice(args.root, args.decode) if args.kind == "practice" else inspect_yolo(args.root, args.hash_images)
    except (OSError, ValueError, RuntimeError, cv2.error, yaml.YAMLError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2))
    unreadable = any(video["frame_count_matches"] is False for video in result.get("videos", {}).values())
    return int(bool(result.get("error_count") or unreadable))


if __name__ == "__main__":
    sys.exit(main())
