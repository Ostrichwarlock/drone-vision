"""Run the detector and causal tracker on a video, camera, or stream."""

import argparse
import logging
from pathlib import Path
import sys

import cv2
import yaml

from src.pipeline.run import Settings, run
from src.trackers import MultiObjectTracker

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Offline input file")
    source.add_argument("--camera", type=int, help="Live OpenCV camera index")
    source.add_argument("--stream", help="Live camera stream URI")
    parser.add_argument("--config", type=Path, help="YAML settings file")
    parser.add_argument("--backend", choices=("onnx", "hailo", "oracle"))
    parser.add_argument("--model", type=Path, help="Custom ONNX or Hailo HEF detector weights")
    parser.add_argument("--annotations", type=Path, help="Oracle detection diagnostic input")
    parser.add_argument("--input-size", type=int, help="Square ONNX detector input size")
    parser.add_argument("--confidence", type=float, help="Detector score threshold")
    parser.add_argument("--iou-threshold", type=float, help="Detector NMS IoU threshold")
    parser.add_argument("--class-id", type=int, help="Drone class index in the model")
    parser.add_argument("--detector-interval", type=int, help="Run detector once per N source frames")
    parser.add_argument("--tiles", type=int, help="Number of tiles per image axis (1 for full frame)")
    parser.add_argument("--tile-overlap", type=float)
    parser.add_argument(
        "--tile-fallback-confidence", type=float,
        help="For ONNX full-frame detection, retry with 2×2 tiles when no box reaches this confidence",
    )
    parser.add_argument("--threads", type=int, help="OpenCV CPU threads")
    parser.add_argument("--max-frames", type=int, help="Stop after processing this many frames")
    parser.add_argument("--expected-frames", type=int, help="Validate offline decoded frame count")
    parser.add_argument("--predictions", type=Path, help="Official-format prediction text file")
    parser.add_argument("--detections", type=Path, help="Raw detector rows on inference frames")
    parser.add_argument("--annotated", type=Path, help="Annotated MP4 output")
    parser.add_argument("--stats", type=Path, help="Processing statistics JSON")
    parser.add_argument("--output-dir", type=Path, help="Create named prediction, video and stats files")
    parser.add_argument("--display", action="store_true", help="Show preview with GUI-enabled OpenCV")
    return parser


def read_config(path: Path | None) -> dict:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.keys() - {"detector", "tracker", "pipeline"}:
        raise ValueError("Config must map detector, tracker and/or pipeline sections")
    if any(not isinstance(section, dict) for section in data.values()):
        raise ValueError("Config sections must be mappings")
    return data


def detector_for(args: argparse.Namespace, config: dict):
    from src.detectors import AnnotatedDetector, OnnxDetector

    options = config.get("detector", {}).copy()
    config_mode = options.pop("backend", None)
    config_model = options.pop("model", None)
    model = args.model or (Path(config_model) if config_model else None)
    mode = args.backend or ("oracle" if args.annotations else config_mode)
    if mode is None:
        mode = "hailo" if model and model.suffix.lower() == ".hef" else "onnx"
    if mode == "oracle":
        if not args.annotations or args.model:
            raise ValueError("Oracle mode requires --annotations and takes no --model")
        LOGGER.warning("Oracle detection diagnostic uses annotated boxes as detections")
        return mode, AnnotatedDetector(args.annotations)
    if args.annotations:
        raise ValueError("--annotations is exclusive to oracle mode")
    if model is None:
        raise ValueError(f"Provide a local model for the {mode} backend with --model")
    if not model.is_file():
        raise ValueError(f"Model file not found: {model}. Set --model to an existing {mode.upper()} model.")
    for name in ("input_size", "confidence", "iou_threshold", "class_id"):
        value = getattr(args, name)
        if value is not None:
            options[name] = value
    if mode == "onnx":
        return mode, OnnxDetector(model, **options)
    if mode == "hailo":
        from src.detectors import HailoDetector

        if "input_size" in options:
            raise ValueError("The HEF specifies Hailo input dimensions, omit --input-size")
        return mode, HailoDetector(model, **options)
    raise ValueError(f"Unknown backend {mode!r}")


def settings_for(args: argparse.Namespace, config: dict, mode: str) -> Settings:
    pipeline = config.get("pipeline", {})
    unknown = pipeline.keys() - {
        "detector_interval", "tiles", "tile_overlap", "tile_fallback_confidence", "threads", "fallback_live_fps",
    }
    if unknown:
        raise ValueError(f"Unknown pipeline settings: {', '.join(sorted(unknown))}")
    interval = args.detector_interval if args.detector_interval is not None else pipeline.get("detector_interval", 1)
    tiles = args.tiles if args.tiles is not None else pipeline.get("tiles", 1)
    overlap = args.tile_overlap if args.tile_overlap is not None else pipeline.get("tile_overlap", 0.20)
    tile_fallback = (
        args.tile_fallback_confidence
        if args.tile_fallback_confidence is not None else pipeline.get("tile_fallback_confidence")
    )
    threads = args.threads if args.threads is not None else pipeline.get("threads")
    fallback_fps = pipeline.get("fallback_live_fps", 30)
    if not isinstance(interval, int) or interval < 1 or not isinstance(tiles, int) or not 1 <= tiles <= 8:
        raise ValueError("Use a positive detector interval and between 1 and 8 tiles per axis")
    if not 0 <= overlap < 1 or not 0 < fallback_fps < 1000:
        raise ValueError("Use tile overlap in [0, 1) and live recording FPS between 0 and 1000")
    if mode == "oracle" and (args.video is None or tiles != 1):
        raise ValueError("Oracle mode accepts offline video with full-frame detections")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.expected_frames is not None and (
        args.video is None or args.expected_frames <= 0 or args.max_frames or args.display
    ):
        raise ValueError("--expected-frames requires an uninterrupted offline run")
    if threads is not None:
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("--threads must be positive")
        cv2.setNumThreads(threads)
    if args.video is not None:
        if not args.video.is_file():
            raise ValueError(f"Missing video file: {args.video}")
        source, source_type = str(args.video), "video"
        stem = args.video.resolve().parent.name if args.video.stem == "video" else args.video.stem
    elif args.camera is not None and args.camera >= 0:
        source, source_type, stem = args.camera, "camera", f"camera-{args.camera}"
    elif args.stream:
        source, source_type, stem = args.stream, "stream", "stream"
    else:
        raise ValueError("Use a nonnegative camera index or a nonempty stream URI")

    predictions, detections, annotated, stats = args.predictions, args.detections, args.annotated, args.stats
    if args.output_dir:
        predictions = predictions or args.output_dir / f"{stem}.txt"
        annotated = annotated or args.output_dir / f"{stem}_annotated.mp4"
        stats = stats or args.output_dir / f"{stem}_stats.json"
    config_model = config.get("detector", {}).get("model")
    protected = {
        Path(path).resolve()
        for path in (args.video, args.model, args.annotations, args.config, config_model) if path
    }
    outputs = [path for path in (predictions, detections, annotated, stats) if path is not None]
    resolved = {path.resolve() for path in outputs}
    if len(outputs) != len(resolved) or protected & resolved:
        raise ValueError("Input and output paths must be distinct")
    if annotated and annotated.suffix.lower() != ".mp4":
        raise ValueError("--annotated must use the .mp4 extension")
    for path in outputs:
        if path.is_dir():
            raise ValueError(f"Output path is a directory: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    return Settings(
        source, source_type, mode, interval=interval, tiles=tiles, tile_overlap=overlap,
        max_frames=args.max_frames, expected_frames=args.expected_frames,
        fallback_live_fps=fallback_fps, predictions=predictions, detections=detections,
        annotated=annotated, stats=stats, display=args.display, tile_fallback_confidence=tile_fallback,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    detector = None
    try:
        config = read_config(args.config)
        mode, detector = detector_for(args, config)
        settings = settings_for(args, config, mode)
        tracker = MultiObjectTracker(**config.get("tracker", {}))
        result = run(settings, detector, tracker)
    except (RuntimeError, OSError, ValueError, TypeError, ImportError, cv2.error, yaml.YAMLError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        close = getattr(detector, "close", None)
        if close is not None:
            close()
    LOGGER.info(
        "%s processed %d frames at %.2f FPS (%.2f source FPS, %d live drops)",
        result["detector_mode"], result["frames"], result["processing_fps"],
        result["source_fps"], result["frames_dropped_live"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
