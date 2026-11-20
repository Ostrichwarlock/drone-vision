"""Repeat a full video-pipeline run and record timing and host telemetry."""

import argparse
from datetime import datetime, timezone
from hashlib import file_digest
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic

PI_FLAGS = {
    0: "undervoltage",
    1: "arm_frequency_capped",
    2: "throttled",
    3: "soft_temperature_limit_active",
    16: "undervoltage_has_occurred",
    17: "arm_frequency_capping_has_occurred",
    18: "throttling_has_occurred",
    19: "soft_temperature_limit_has_occurred",
}


def read_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").rstrip("\x00\n").strip()
    except OSError:
        return None


def host_details() -> dict:
    model = read_file(Path("/proc/device-tree/model")) or read_file(Path("/sys/firmware/devicetree/base/model"))
    release = platform.platform()
    if sys.platform == "linux":
        try:
            release = platform.freedesktop_os_release().get("PRETTY_NAME", release)
        except OSError:
            pass
    try:
        memory_mib = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024))
        if memory_mib <= 0:
            memory_mib = None
    except (OSError, ValueError):
        memory_mib = None
    return {
        "system": platform.system(),
        "release": release,
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "model": model,
        "ram_mib": memory_mib,
        "python": platform.python_version(),
        "raspberry_pi": bool(model and model.startswith("Raspberry Pi")),
        "hailort_cli_version": firmware_value(shutil.which("hailortcli"), "--version") or None,
    }


def cpu_sensor() -> Path | None:
    for item in Path("/sys/class/thermal").glob("thermal_zone*"):
        kind = read_file(item / "type")
        if kind and ("cpu" in kind.lower() or kind.lower() == "x86_pkg_temp"):
            return item / "temp"
    return None


def firmware_value(utility: str | None, command: str) -> str:
    if utility is None:
        return ""
    try:
        result = subprocess.run([utility, command], capture_output=True, text=True, timeout=0.5, check=True)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def telemetry_sample(sensor: Path | None, firmware: str | None, start: float) -> dict:
    temperature = None
    temp_data = read_file(sensor) if sensor else None
    if temp_data and temp_data.lstrip("-").isdigit():
        temperature = int(temp_data) / 1000.0
    else:
        temp_match = re.match(r"temp=(-?[0-9]+(?:\.[0-9]+)?)", firmware_value(firmware, "measure_temp"))
        if temp_match:
            temperature = float(temp_match.group(1))
    flag_match = re.fullmatch(r"throttled=(0x[0-9a-fA-F]+)", firmware_value(firmware, "get_throttled"))
    bitmask = int(flag_match.group(1), 16) if flag_match else None
    return {
        "elapsed_seconds": round(monotonic() - start, 3),
        "cpu_temperature_c": temperature,
        "throttling_flags_hex": None if bitmask is None else hex(bitmask),
        "throttling_flags": None if bitmask is None else [
            label for bit, label in PI_FLAGS.items() if bitmask & (1 << bit)
        ],
    }


def summarize_telemetry(samples: list[dict]) -> dict:
    temperatures = [row["cpu_temperature_c"] for row in samples if row["cpu_temperature_c"] is not None]
    flag_sets = [row["throttling_flags"] for row in samples if row["throttling_flags"] is not None]
    return {
        "cpu_temperature_peak_c": max(temperatures, default=None),
        "throttling_flags_observed": sorted({flag for flags in flag_sets for flag in flags}) if flag_sets else None,
        "samples": samples,
    }


def run_once(
    command: list[str],
    stats_path: Path,
    sample_seconds: float,
    sensor: Path | None,
    firmware: str | None,
) -> dict:
    started = monotonic()
    samples = [telemetry_sample(sensor, firmware, started)] if sensor or firmware else []
    with subprocess.Popen(command) as child:
        if not samples:
            child.wait()
        while child.poll() is None:
            try:
                child.wait(timeout=sample_seconds)
            except subprocess.TimeoutExpired:
                samples.append(telemetry_sample(sensor, firmware, started))
        if child.returncode:
            raise RuntimeError(f"Video pipeline exited with code {child.returncode}")
    if samples:
        samples.append(telemetry_sample(sensor, firmware, started))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return {"pipeline": stats, "device": summarize_telemetry(samples)}


def report_args(arguments: list[str]) -> list[str]:
    result = []
    redact_next = False
    for argument in arguments:
        if redact_next:
            result.append("<live stream>")
        elif argument.startswith("--stream="):
            result.append("--stream=<live stream>")
        else:
            result.append(argument)
        redact_next = argument == "--stream"
    return result


def sha256_for(path: Path | None) -> str | None:
    if path is None:
        return None
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Benchmark report JSON")
    parser.add_argument("--repeat", type=int, default=1, help="Number of fresh-process runs")
    parser.add_argument("--render", action="store_true", help="Render and encode to a temporary MP4 for each run")
    parser.add_argument("--sample-seconds", type=float, default=1.0, help="Host telemetry sampling period")
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER, metavar="-- RUN_VIDEO_OPTIONS")
    args = parser.parse_args()
    arguments = args.pipeline_args[1:] if args.pipeline_args[:1] == ["--"] else args.pipeline_args
    if not arguments or args.repeat < 1 or not math.isfinite(args.sample_seconds) or args.sample_seconds < 0.1:
        parser.error("Use positive repetitions, a sampling period of at least 0.1 seconds, and pass run options after --")
    output_path = args.output.resolve()
    from scripts.run_video import build_parser, read_config, settings_for

    try:
        run_parser = build_parser()
        run_parser.allow_abbrev = False
        run_args = run_parser.parse_args(arguments)
        if run_args.stats is not None:
            raise ValueError("Use the benchmark's --output instead of passing --stats")
        if args.repeat > 1 and any((run_args.predictions, run_args.detections, run_args.annotated, run_args.output_dir)):
            raise ValueError("Repeat benchmarks without explicit pipeline outputs. Use --render to include video encoding.")
        if args.render and (run_args.annotated or run_args.output_dir):
            raise ValueError("--render generates its own annotated video")
        inputs = (run_args.video, run_args.model, run_args.annotations, run_args.config)
        config = read_config(run_args.config)
        model = config.get("detector", {}).get("model")
        if model is not None:
            inputs += (Path(model),)
        # Resolve the pipeline's output-directory defaults before starting it.
        mode = "oracle" if run_args.annotations else run_args.backend or "onnx"
        settings = settings_for(run_args, config, mode)
        paths = (*inputs, settings.predictions, settings.detections, settings.annotated, settings.stats)
        if any(path is not None and path.resolve() == output_path for path in paths):
            raise ValueError("Benchmark report must have a path distinct from all pipeline inputs and outputs")
        host = host_details()
        model_path = None if mode == "oracle" else run_args.model or (Path(model) if model else None)
        model_hash = sha256_for(model_path)
        config_hash = sha256_for(run_args.config)
        sensor = cpu_sensor()
        firmware = shutil.which("vcgencmd") if host["raspberry_pi"] else None
        runs = []
        with TemporaryDirectory(prefix="drone-benchmark-") as directory:
            for index in range(1, args.repeat + 1):
                stats = Path(directory) / f"run-{index}.json"
                command = [sys.executable, "-m", "scripts.run_video", *arguments, "--stats", str(stats)]
                video = Path(directory) / f"run-{index}.mp4"
                if args.render:
                    command.extend(["--annotated", str(video)])
                runs.append(run_once(command, stats, args.sample_seconds, sensor, firmware))
                if args.render:
                    video.unlink()
        report = {
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": host,
            "run_video_arguments": report_args(arguments),
            "model_sha256": model_hash,
            "config_sha256": config_hash,
            "mode": "visualized" if args.render or settings.annotated or settings.display else "headless",
            "median_processing_fps": median(run["pipeline"]["processing_fps"] for run in runs),
            "accelerator_utilization_percent": None,
            "runs": runs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{len(runs)} run(s), median {report['median_processing_fps']:.2f} processing FPS")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
