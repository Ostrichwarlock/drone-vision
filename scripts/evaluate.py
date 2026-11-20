"""Evaluate one offline video against MOT-style ground truth."""

import argparse
import json
from pathlib import Path
import sys

from src.evaluation import evaluate
from src.mot import read_mot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--detections", type=Path, help="Optional raw detector rows exported by run_video")
    parser.add_argument("--run-stats", type=Path, help="run_video JSON, including detector mode and performance")
    parser.add_argument("--output", type=Path, help="Write the JSON report here")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-gap-frames", type=int, default=1)
    parser.add_argument("--recovery-window-frames", type=int, default=15)
    parser.add_argument(
        "--ground-truth-offset", type=float, default=0.0,
        help="Shift annotated left and top coordinates, for example -1 for one-based pixel coordinates",
    )
    args = parser.parse_args()

    try:
        inputs = [path for path in (args.ground_truth, args.predictions, args.detections, args.run_stats) if path]
        if args.output and args.output.resolve() in {path.resolve() for path in inputs}:
            raise ValueError("Evaluation output must have a path distinct from all inputs")
        truth = read_mot(args.ground_truth, ground_truth=True, coordinate_offset=args.ground_truth_offset)
        predictions = read_mot(args.predictions)
        stats = json.loads(args.run_stats.read_text(encoding="utf-8")) if args.run_stats else None
        if stats is not None and not isinstance(stats, dict):
            raise ValueError("Run stats must contain a JSON object")
        mode = stats.get("detector_mode") if stats else None
        detections = None
        if args.detections and mode != "oracle":
            detections = read_mot(args.detections, require_unique_ids=False)
        if stats and stats.get("source_type") == "video":
            last_frame = stats.get("last_source_frame")
            if isinstance(last_frame, int) and max(truth.keys() | predictions.keys() | (detections or {}).keys(), default=0) > last_frame:
                raise ValueError("Evaluation rows extend beyond the last processed source frame")
        result = evaluate(
            truth, predictions, detections,
            iou_threshold=args.iou_threshold,
            min_gap_frames=args.min_gap_frames,
            recovery_window_frames=args.recovery_window_frames,
        )
        result["detector_mode"] = mode
        result["ground_truth_offset"] = args.ground_truth_offset
        result["performance"] = stats
        if mode == "oracle":
            result["scope"] = "oracle_detection_diagnostic"
        elif mode in ("onnx", "hailo"):
            result["scope"] = "inference"
        else:
            result["scope"] = "detector_provenance_unspecified"
        report = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
        else:
            print(report, end="")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
