"""Fine-tune a local Ultralytics YOLOv8 or YOLO11 detector on a YOLO dataset."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML with train and val splits")
    parser.add_argument("--model", type=Path, required=True, help="Local starting YOLO .pt weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", help="Training device, such as mps, cpu or 0")
    parser.add_argument("--project", type=Path, default=Path("outputs/train"))
    parser.add_argument("--name", default="drone")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.data.suffix.lower() not in {".yaml", ".yml"} or not args.data.is_file():
        parser.error("--data must point to a local YOLO dataset YAML")
    if args.model.suffix.lower() != ".pt" or not args.model.is_file():
        parser.error("--model must point to local YOLO .pt weights")
    if args.imgsz < 32 or args.imgsz % 32 or args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("Use a multiple of 32 for --imgsz, a positive epoch and batch count, and nonnegative workers")

    try:
        from ultralytics import YOLO
    except ImportError:
        parser.exit(1, "Install Ultralytics in the workstation training environment.\n")
    model = YOLO(str(args.model.resolve()))
    if model.task != "detect":
        parser.error("--model must be a detection checkpoint")
    model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        seed=args.seed,
    )
    print(f"Best checkpoint: {model.trainer.best}")


if __name__ == "__main__":
    main()
