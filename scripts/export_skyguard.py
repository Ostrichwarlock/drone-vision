"""Export the verified SkyGuard YOLO11n checkpoint to static 640-pixel ONNX."""

import argparse
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from ultralytics.engine.exporter import Exporter
from ultralytics.nn.tasks import DetectionModel


CHECKPOINT_SHA256 = "33f5bc2309b61427ee77a00110c0b01cf7f03912b3f4f2fa65ca2793c2550f90"
MODEL_GLOBALS = {
    "torch.nn.modules.activation.SiLU",
    "torch.nn.modules.batchnorm.BatchNorm2d",
    "torch.nn.modules.container.ModuleList",
    "torch.nn.modules.container.Sequential",
    "torch.nn.modules.conv.Conv2d",
    "torch.nn.modules.linear.Identity",
    "torch.nn.modules.pooling.MaxPool2d",
    "torch.nn.modules.upsampling.Upsample",
    "ultralytics.nn.modules.block.Attention",
    "ultralytics.nn.modules.block.Bottleneck",
    "ultralytics.nn.modules.block.C2PSA",
    "ultralytics.nn.modules.block.C3k",
    "ultralytics.nn.modules.block.C3k2",
    "ultralytics.nn.modules.block.DFL",
    "ultralytics.nn.modules.block.PSABlock",
    "ultralytics.nn.modules.block.SPPF",
    "ultralytics.nn.modules.conv.Concat",
    "ultralytics.nn.modules.conv.Conv",
    "ultralytics.nn.modules.conv.DWConv",
    "ultralytics.nn.modules.head.Detect",
    "ultralytics.nn.tasks.DetectionModel",
}


class _StateNode:
    """Represent stored module state without reconstructing executable modules."""


def _tensors(node: _StateNode, prefix: str = "") -> dict[str, torch.Tensor]:
    if type(node) is not _StateNode:
        raise ValueError("Unexpected checkpoint module")
    state = {}
    for name, value in vars(node).get("_parameters", {}).items():
        if value is not None:
            if type(value) not in {torch.nn.Parameter, torch.Tensor}:
                raise ValueError("Unexpected checkpoint parameter")
            state[prefix + name] = value.detach()
    excluded = vars(node).get("_non_persistent_buffers_set", set())
    for name, value in vars(node).get("_buffers", {}).items():
        if value is not None and name not in excluded:
            if type(value) is not torch.Tensor:
                raise ValueError("Unexpected checkpoint buffer")
            state[prefix + name] = value
    for name, module in vars(node).get("_modules", {}).items():
        if module is not None:
            state.update(_tensors(module, prefix + name + "."))
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/skyguard-yolo11n.onnx"))
    args = parser.parse_args()
    if args.output.suffix.lower() != ".onnx":
        parser.error("--output must end in .onnx")

    with args.checkpoint.open("rb") as checkpoint:
        if file_digest(checkpoint, "sha256").hexdigest() != CHECKPOINT_SHA256:
            parser.error("The checkpoint does not match the verified SkyGuard SHA-256")
        checkpoint.seek(0)
        if set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)) != MODEL_GLOBALS:
            parser.error("The checkpoint has unexpected serialized types")
        checkpoint.seek(0)
        with torch.serialization.safe_globals([(_StateNode, name) for name in sorted(MODEL_GLOBALS)]):
            package = torch.load(checkpoint, map_location="cpu", weights_only=True)

    serialized = package.get("ema") or package["model"]
    model = DetectionModel("yolo11n.yaml", nc=1, verbose=False)
    model.load_state_dict(_tensors(serialized), strict=True)
    model.task = "detect"
    model.names = {0: "drone"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix="skyguard-export-") as directory:
        model.pt_path = str(Path(directory).resolve() / "skyguard-yolo11n.pt")
        exporter = Exporter(overrides={
            "format": "onnx", "imgsz": 640, "opset": 12, "batch": 1,
            "dynamic": False, "nms": False, "simplify": True, "device": "cpu",
        })
        Path(exporter(model)).replace(args.output)
    with args.output.open("rb") as stream:
        print(f"{file_digest(stream, 'sha256').hexdigest()}  {args.output}")


if __name__ == "__main__":
    main()
