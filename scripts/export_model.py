"""Export local YOLOv8 or YOLO11 weights as a static OpenCV CPU ONNX model."""

import argparse
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Local drone detector .pt weights")
    parser.add_argument("--output", type=Path, required=True, help="Destination .onnx file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()
    if args.weights.suffix.lower() != ".pt" or not args.weights.is_file():
        parser.error("--weights must point to a local detection .pt checkpoint")
    if args.output.suffix.lower() != ".onnx":
        parser.error("--output must be an .onnx file")
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error("--imgsz must be a positive multiple of 32")

    try:
        from ultralytics import YOLO
    except ImportError:
        parser.exit(1, "Install Ultralytics in the workstation training environment.\n")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".drone-export-") as scratch:
        weights = Path(scratch) / args.weights.name
        copyfile(args.weights, weights)
        model = YOLO(str(weights))
        if model.task != "detect" or getattr(model.model, "end2end", False):
            parser.error("--weights must be a standard YOLOv8 or YOLO11 detection checkpoint")
        exported = Path(model.export(
            format="onnx", imgsz=args.imgsz, opset=args.opset, batch=1,
            dynamic=False, nms=False, simplify=True, device="cpu",
        ))
        exported.replace(output)
    print(f"Exported {output} ({args.imgsz} × {args.imgsz})")


if __name__ == "__main__":
    main()
