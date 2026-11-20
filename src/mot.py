"""Read MOT-style boxes and normalize their coordinates for internal use."""

from collections import defaultdict
import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from src.types import Box


@dataclass(frozen=True, slots=True)
class MotRow:
    id: int
    box: Box
    confidence: float


def read_mot(
    path: Path,
    *,
    ground_truth: bool = False,
    coordinate_offset: float = 0.0,
    require_unique_ids: bool = True,
) -> dict[int, list[MotRow]]:
    """Read headerless 1-indexed rows, optionally allowing unlabelled detections."""
    path = Path(path)
    if not isfinite(coordinate_offset):
        raise ValueError("MOT coordinate offset must be finite")
    frames: dict[int, list[MotRow]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for number, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not field.strip() for field in row) or row[0].lstrip().startswith("#"):
                continue
            try:
                if (ground_truth and len(row) < 6) or (not ground_truth and len(row) != 7):
                    raise ValueError("expected MOT ground-truth fields" if ground_truth else "expected seven fields")
                frame, identity = map(int, row[:2])
                left, top, width, height = map(float, row[2:6])
                confidence = float(row[6]) if len(row) > 6 else 1.0
                left += coordinate_offset
                top += coordinate_offset
                box = (left, top, left + width, top + height)
                if frame < 1 or ((require_unique_ids or ground_truth) and identity < 1):
                    raise ValueError("frames and track IDs must be positive")
                if not all(map(isfinite, (*box, confidence))) or not (width > 0 and height > 0):
                    raise ValueError("expected finite coordinates, score, and a positive box size")
                if ground_truth and confidence == 0:
                    continue
                if ground_truth and confidence == -1:
                    confidence = 1.0
                if not 0 <= confidence <= 1:
                    raise ValueError("confidence must be in [0, 1]")
                key = (frame, identity)
                if require_unique_ids:
                    if key in seen:
                        raise ValueError(f"duplicate ID {identity} on frame {frame}")
                    seen.add(key)
                frames[frame].append(MotRow(identity, box, confidence))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return {frame: sorted(rows, key=lambda item: item.id) for frame, rows in frames.items()}
