"""Run and score all annotated video sequences in an organizer dataset."""

import argparse
import csv
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from src.evaluation.metrics import RECOVERY_BUCKETS, object_counts, rate, recovery_summary, tracking_accuracy
from src.mot import read_mot


def find_sequences(dataset: Path, *, scoring_only: bool) -> list[Path]:
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset}")
    if scoring_only:
        annotations = sorted(dataset.rglob("gt/gt.txt"))
        if not annotations:
            raise ValueError(f"No gt/gt.txt annotations found under {dataset}")
        return [path.parent.parent for path in annotations]
    videos = sorted(path for path in dataset.rglob("video.mp4") if path.is_file())
    if not videos:
        raise ValueError(f"No sequence video.mp4 files found under {dataset}")
    missing = [video.parent / "gt" / "gt.txt" for video in videos if not (video.parent / "gt" / "gt.txt").is_file()]
    if missing:
        raise ValueError(f"Missing ground truth for: {', '.join(str(path) for path in missing)}")
    return [video.parent for video in videos]


def find_predictions(root: Path, sequence: Path, relative: Path) -> Path:
    for path in (root / relative / f"{sequence.name}.txt", root / f"{sequence.name}.txt"):
        if path.is_file():
            return path.resolve()
    raise ValueError(f"Missing prediction for {relative}: expected {sequence.name}.txt under {root}")


def run_stage(module: str, arguments: list[str], sequence: Path) -> None:
    try:
        subprocess.run([sys.executable, "-m", module, *arguments], check=True)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"{sequence}: {module} exited with status {error.returncode}") from error


def prepare_oracle_truth(source: Path, output: Path, offset: float) -> Path:
    """Move one-based MOT coordinates into the frame coordinates used by replay."""
    if offset == 0:
        return source
    annotations = read_mot(source, ground_truth=True, coordinate_offset=offset)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for frame, rows in annotations.items():
            for row in rows:
                left, top, right, bottom = row.box
                writer.writerow([frame, row.id, left, top, right - left, bottom - top, row.confidence])
    return output


def run_sequence(sequence: Path, relative: Path, directory: Path, args: argparse.Namespace) -> Path:
    truth = sequence / "gt" / "gt.txt"
    predictions = directory / f"{sequence.name}.txt"
    detections = directory / f"{sequence.name}_raw.txt"
    stats = directory / f"{sequence.name}_stats.json"

    options = ["--video", str(sequence / "video.mp4"), "--predictions", str(predictions), "--stats", str(stats)]
    for name in ("config", "class_id", "threads"):
        value = getattr(args, name)
        if value is not None:
            options.extend([f"--{name.replace('_', '-')}", str(value)])
    if args.render:
        options.extend(["--annotated", str(directory / f"{sequence.name}_annotated.mp4")])
    if args.oracle:
        replay = prepare_oracle_truth(truth, directory / ".replay.txt", args.ground_truth_offset)
        options.extend(["--backend", "oracle", "--annotations", str(replay)])
    else:
        options.extend(["--model", str(args.model), "--detections", str(detections)])
        if args.backend:
            options.extend(["--backend", args.backend])

    run_stage("scripts.run_video", options, relative)
    if args.oracle and replay != truth:
        replay.unlink()
    return predictions


def evaluate_sequence(
    sequence: Path,
    relative: Path,
    staging: Path,
    args: argparse.Namespace,
    predictions: Path | None,
) -> dict:
    directory = staging / relative
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {relative}", file=sys.stderr, flush=True)
    if predictions is None:
        predictions = run_sequence(sequence, relative, directory, args)

    evaluation = directory / "evaluation.json"
    stats = predictions.with_name(f"{predictions.stem}_stats.json")
    detections = predictions.with_name(f"{predictions.stem}_raw.txt")
    options = [
        "--ground-truth", str(sequence / "gt" / "gt.txt"),
        "--predictions", str(predictions),
        "--iou-threshold", str(args.iou_threshold),
        "--ground-truth-offset", str(args.ground_truth_offset),
        "--min-gap-frames", str(args.min_gap_frames),
        "--recovery-window-frames", str(args.recovery_window_frames),
        "--output", str(evaluation),
    ]
    if stats.is_file():
        options.extend(["--run-stats", str(stats)])
    if detections.is_file():
        options.extend(["--detections", str(detections)])
    run_stage("scripts.evaluate", options, relative)
    return json.loads(evaluation.read_text(encoding="utf-8"))


def aggregate_counts(sections: list[dict]) -> dict:
    matches = sum(section["true_positives"] for section in sections)
    truth = matches + sum(section["false_negatives"] for section in sections)
    predictions = matches + sum(section["false_positives"] for section in sections)
    return object_counts(matches, truth, predictions)


def aggregate_reports(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("A dataset summary requires at least one sequence")
    scopes = {report["scope"] for report in reports}
    if "oracle_detection_diagnostic" in scopes and len(scopes) > 1:
        raise ValueError("Oracle and model or unknown predictions require separate dataset reports")
    scope = next(iter(scopes)) if len(scopes) == 1 else "detector_provenance_unspecified"
    detection = [report["detections"] for report in reports if report["detections"] is not None]
    tracking = [report["tracking"] for report in reports]
    tracking_counts = aggregate_counts(tracking)
    id_switches = sum(report["id_switches"] for report in tracking)
    idtp, idfp, idfn = (sum(part[key] for part in tracking) for key in ("idtp", "idfp", "idfn"))
    events = [event for report in reports for event in report["recovery"]["details"]]
    timings = [report["performance"] for report in reports]
    performance = None
    if all(timing and {"frames", "elapsed_seconds"} <= timing.keys() for timing in timings):
        frames = sum(timing["frames"] for timing in timings)
        elapsed = sum(timing["elapsed_seconds"] for timing in timings)
        performance = {"frames": frames, "elapsed_seconds": elapsed, "processing_fps": rate(frames, elapsed)}
    return {
        "scope": scope,
        "detections": aggregate_counts(detection) if len(detection) == len(reports) else None,
        "tracking": {
            **tracking_counts,
            "idtp": idtp,
            "idfp": idfp,
            "idfn": idfn,
            "id_precision": rate(idtp, idtp + idfp),
            "id_recall": rate(idtp, idtp + idfn),
            "idf1": rate(2 * idtp, 2 * idtp + idfp + idfn),
            "id_switches": id_switches,
            "mota": tracking_accuracy(
                idtp + idfn, tracking_counts["false_positives"], tracking_counts["false_negatives"], id_switches,
            ),
        },
        "recovery": {
            **recovery_summary(events),
            "by_gap_frames": [
                {
                    "min": low,
                    "max": high,
                    **recovery_summary([
                        event for event in events if event["gap_frames"] >= low
                        and (high is None or event["gap_frames"] <= high)
                    ]),
                }
                for low, high in RECOVERY_BUCKETS
            ],
        },
        "performance": performance,
    }


def print_submission_table(sequences: list[dict], aggregate: dict, iou_threshold: float) -> None:
    scope = aggregate["scope"]
    title = {
        "oracle_detection_diagnostic": "Oracle tracking diagnostic",
        "detector_provenance_unspecified": "Submission scores (detector provenance unspecified)",
    }.get(scope, "Submission scores")
    print(f"{title} at IoU {iou_threshold:g}. Metrics use tracked boxes.")
    columns = (
        ("Detection F1", "f1"), ("Precision", "precision"), ("Recall", "recall"),
        ("IDF1", "idf1"), ("ID switches", "id_switches"), ("MOTA", "mota"),
    )
    rows = [("Sequence", *(label for label, _ in columns))]
    scored = [(part["sequence"], part["tracking"]) for part in sequences]
    for sequence, tracking in [*scored, ("OVERALL", aggregate["tracking"])]:
        values = []
        for _, key in columns:
            value = tracking[key]
            values.append("n/a" if value is None else str(value) if key == "id_switches" else f"{value:.4f}")
        rows.append((sequence, *values))
    widths = [max(map(len, column)) for column in zip(*rows)]
    for label, *values in rows:
        print("  ".join([label.ljust(widths[0]), *(value.rjust(width) for value, width in zip(values, widths[1:]))]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Root containing sequence folders with gt/gt.txt")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory for results")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="Local ONNX or Hailo HEF detector")
    source.add_argument("--oracle", action="store_true", help="Replay ground-truth boxes to diagnose tracking only")
    source.add_argument("--pred-dir", type=Path, help="Score existing <sequence>.txt submissions without running video inference")
    parser.add_argument("--config", type=Path, help="Runtime settings passed to run_video")
    parser.add_argument("--backend", choices=("onnx", "hailo"), help="Override the detector backend")
    parser.add_argument("--class-id", type=int, help="Model class index for drones")
    parser.add_argument("--threads", type=int, help="OpenCV CPU threads")
    parser.add_argument("--render", action="store_true", help="Include annotated MP4s and their writing time")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--ground-truth-offset", type=float, default=0.0,
                        help="Shift annotated left and top, for example -1 for one-based pixel coordinates")
    parser.add_argument("--min-gap-frames", type=int, default=1)
    parser.add_argument("--recovery-window-frames", type=int, default=15)
    args = parser.parse_args()
    if args.oracle and (args.backend or args.class_id is not None):
        parser.error("--backend and --class-id apply only to model inference")
    if args.pred_dir and (args.backend or args.class_id is not None or args.config or args.threads is not None or args.render):
        parser.error("--pred-dir scores saved tracks and does not take runtime configuration")
    if not 0 < args.iou_threshold <= 1 or args.min_gap_frames < 1 or args.recovery_window_frames < 1:
        parser.error("Use an IoU threshold in (0, 1] and positive frame counts")
    if not isfinite(args.ground_truth_offset):
        parser.error("--ground-truth-offset must be finite")

    try:
        dataset = args.dataset.resolve()
        paths = find_sequences(dataset, scoring_only=args.pred_dir is not None)
        relatives = [path.relative_to(dataset) if path != dataset else Path(dataset.name) for path in paths]
        predictions = [None] * len(paths)
        if args.pred_dir:
            predictions = [find_predictions(args.pred_dir, path, relative) for path, relative in zip(paths, relatives)]
            if len(set(predictions)) != len(predictions):
                raise ValueError("Multiple sequences resolve to the same prediction file")
        output = args.output_dir.resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"Use a new or empty output directory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".drone-evaluation-", dir=output.parent) as temporary:
            staging = Path(temporary) / "results"
            sequences = []
            reports = []
            for path, relative, prediction in zip(paths, relatives, predictions):
                report = evaluate_sequence(path, relative, staging, args, prediction)
                reports.append(report)
                sequences.append({
                    "sequence": relative.as_posix(),
                    "report": (relative / "evaluation.json").as_posix(),
                    "detections": report["detections"],
                    "tracking": report["tracking"],
                    "recovery": recovery_summary(report["recovery"]["details"]),
                    "performance": report["performance"],
                })
            summary = {
                "sequence_count": len(sequences),
                "iou_threshold": args.iou_threshold,
                "ground_truth_offset": args.ground_truth_offset,
                "min_gap_frames": args.min_gap_frames,
                "recovery_window_frames": args.recovery_window_frames,
                "submission_metrics_source": "tracking",
                "aggregate": aggregate_reports(reports),
                "sequences": sequences,
            }
            (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            staging.replace(output)
        print_submission_table(sequences, summary["aggregate"], args.iou_threshold)
        print(f"Wrote {output / 'summary.json'}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
