# Raspberry Pi 5 deployment

The target specified by the challenge is a Raspberry Pi 5 with 16 GB RAM and a Hailo accelerator. The exact board and chip variant need to be confirmed on the target.

## Identify the device

Run on the Pi using its existing operating system and driver installation.

```bash
uname -a
uname -m
cat /etc/os-release
free -h
lscpu
df -h .
lspci -nn
hailortcli scan
hailortcli fw-control identify
hailortcli --version
/usr/bin/python3 -c 'import sys, hailo_platform; print(sys.version); print(hailo_platform.__file__)'
rpicam-hello --list-cameras
```

Check for a 64-bit `aarch64` OS, record the system Python and HailoRT versions, and read the Hailo architecture from device identification. Missing commands indicate the corresponding tool is unavailable. If the driver or runtime is missing, follow the instructions for the board actually installed. The official AI Kit and AI HAT+ use `hailo-all`. The AI HAT+ 2 uses `hailo-h10-all`. Choose the documented runtime and firmware versions for your board and OS. [Raspberry Pi AI software guide](https://www.raspberrypi.com/documentation/computers/ai.html)

## Set up the project environment

Copy the workspace and the chosen model to the Pi. Create a project venv with the system interpreter so the installed HailoRT bindings are available to the pipeline.

```bash
uv venv --python /usr/bin/python3 --system-site-packages
uv sync --locked --python /usr/bin/python3 --no-dev --no-build
uv run --no-sync python -c 'import cv2, numpy, scipy, hailo_platform; print(cv2.__version__, numpy.__version__, hailo_platform.__file__)'
```

The venv creation command is for a fresh checkout. Reuse an existing Pi environment if it already uses the system interpreter and exposes the HailoRT bindings. The project runtime was resolved against Linux ARM64 wheels for Python 3.11 and 3.13. The native HailoRT ABI and device access still need to be verified on the board. [uv project configuration](https://docs.astral.sh/uv/concepts/projects/config/)

## Compile and check the HEF

Use a compatible Linux x86_64 workstation with the matching Dataflow Compiler. The Hailo-8 and Hailo-8L toolchain uses DFC 3.x and Hailo Model Zoo 2.x. Hailo-10H uses DFC 5.x. Confirm the exact version combinations for the chosen toolchain and installed HailoRT. A YOLO11n or YOLOv8n standard detection head with class 0 set to drone is a starting point to validate. Use representative training or validation images for calibration. [Ultralytics Hailo export](https://docs.ultralytics.com/integrations/hailo), [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)

```bash
# Set HAILO_TARGET to the confirmed architecture accepted by the exporter.
# Set YOLO_CHECKPOINT and CALIBRATION_YAML to files on the compilation workstation.
yolo export model="$YOLO_CHECKPOINT" format=hailo \
  name="$HAILO_TARGET" imgsz=640 data="$CALIBRATION_YAML" conf=0.10 iou=0.45
```

The Python backend expects an RGB UINT8 detection HEF with an integrated per-class NMS output. The compiled confidence threshold determines which proposals reach Python. Confirm the model architecture and I/O layout on the Pi before running the pipeline. Set `detector.expected_arch` in the Pi YAML to `hailo8`, `hailo8l`, or `hailo10h` to have the application check the attached device as well.

```bash
hailortcli parse-hef models/drone.hef
hailortcli run models/drone.hef
uv run --no-sync python -m scripts.run_video --config configs/pi5_hailo.yaml \
  --model models/drone.hef --video datasets/practice/SEQUENCE/video.mp4 \
  --max-frames 60 --output-dir outputs/pi-smoke
```

The CLI smoke test checks that the runtime accepts the model. The project smoke test exercises preprocessing, inference, and output on real video. The Hailo Apps examples can be used to check the installed NPU independently. [Hailo Apps](https://github.com/hailo-ai/hailo-apps), [Hailo Raspberry Pi examples](https://github.com/hailo-ai/hailo-rpi5-examples)

## Measure performance

Benchmark CPU ONNX and Hailo HEF with equivalent model weights and input sizes. Run on the hard video, starting with headless mode. Run a separate annotated pass to measure rendering and encoding overhead.

```bash
uv run --no-sync python -m scripts.benchmark --output outputs/pi-hailo.json --repeat 3 -- \
  --video datasets/practice/SEQUENCE/video.mp4 \
  --config configs/pi5_hailo.yaml --model models/drone.hef
uv run --no-sync python -m scripts.benchmark --output outputs/pi-onnx.json --repeat 3 -- \
  --video datasets/practice/SEQUENCE/video.mp4 \
  --config configs/default.yaml --model models/drone.onnx --threads 4
```

Record NPU utilization separately if your HailoRT version exposes it. For Hailo-8/8L, run `hailortcli monitor` in a second terminal and prefix the Hailo benchmark command with `HAILO_MONITOR=1`. Check the monitor help on the installed runtime for other chips. Compare the same configurations with tracking metrics and pay special attention to the longest disappearance. [Hailo-8/8L monitor source](https://github.com/hailo-ai/hailort/blob/hailo8/hailort/hailortcli/mon_command.cpp)
