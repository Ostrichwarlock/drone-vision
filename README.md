# Drone Vision

This prototype detects drones from a video or live camera, tracks them with a Kalman filter, and uses appearance and motion history to recover global IDs after re-entry. It processes files sequentially and emits competition-format tracking text, annotated MP4, and timing data.

## Architecture at a glance

```text
Video, camera, or stream
  → frame capture and timestamps
  → drone boxes from ONNX, Hailo HEF, or annotation replay
  → Kalman motion tracking and re-identification
  → tracked boxes with global IDs
  → annotated MP4, prediction TXT, and timing JSON
```

- **CPU models.** `run_video` runs 640×640 ONNX exports. SkyGuard YOLO11n with **Higher recall** scored the best detection F1 on the organizer practice videos. CERBER-CV-v2 with **Balanced** scored the best IDF1 of the three runs.
- **Browser test console.** `run_ui` accepts a video and calls the same `run_video` pipeline. The browser shows the annotated result, downloads, and scores from `evaluate` when ground truth is available.
- **Synthetic tracker test.** `make_demo` generates frames and exact boxes. `run_video` tracks those boxes, and `evaluate` checks whether it recovers the previous ID.
- **Custom detector.** `inspect_dataset → train → export_model → run_video` produces an ONNX model and runs it on CPU. On Hailo, compile the model to a HEF for your chip and run it through the same tracker.

[Full architecture and script order](#architecture-reference)

## Set up with uv

Run these commands from the project directory. uv uses Python 3.11 by default and recreates the development environment from `uv.lock`.

```bash
uv sync --locked --no-build
```

`requirements-pi.txt` and `requirements-dev.txt` are exports of the lockfile for installations that use requirements files. The Pi runtime uses NumPy, SciPy, PyYAML, and headless OpenCV. The Hailo backend additionally uses the HailoRT Python bindings supplied with the matching Raspberry Pi runtime. See the [Pi 5 setup guide](docs/raspberry_pi_setup.md).

## Test a video in the browser

Start the local console from the project directory.

```bash
uv run --locked --group ui python -m scripts.run_ui
```

Keep the terminal running and open the console on the same machine. The browser UI uses CPU inference and keeps uploaded files and results on your machine. [Open the test console](http://127.0.0.1:8000)

If port 8000 is in use, append `--port 8001` to the launch command and open the console on that port. [Open the test console on port 8001](http://127.0.0.1:8001)

1. Choose or drop an MP4, MOV, AVI, MKV, WebM, or M4V up to 512 MiB and select a detector. For practice footage, select **SkyGuard YOLO11n** with **Higher recall** to use the setup with the best detection F1. The console lists models present in `models/`. [Model setup](#run-the-cpu-quickstart)
2. Choose **Higher recall** to retry with overlapping tiles on frames where the full-frame detector fails to find a box at confidence 0.25. This setting also lets tracks start at confidence 0.25 and can take longer. Optionally add matching MOT annotations, then click **Run video test**. **Run synthetic demo** uses generated frames and annotated boxes to test tracking and ID recovery.
3. Play the annotated video and review frame count, processing FPS, track observations, and recovered IDs. Download the available video, prediction rows, detector rows, and JSON reports from the run.
4. Open **Run statistics** below the annotated video to see source resolution and FPS, model input size when available, runtime, inference calls, CPU usage, and peak memory. Expand the timing table for per-stage latency in milliseconds. The panel appears on every completed run, including uploads without ground truth.
5. Review **Evaluation** for the six team scores under **Submission scores**, based on the saved tracking predictions. Raw detector scores and ID recovery are shown separately. You can attach ground truth to a completed upload and evaluate its saved results.

Ground truth is a headerless MOT-style TXT or CSV file up to 10 MiB, with one row per visible object per frame. The required fields are `frame,id,left,top,width,height`. Use 1-based frame numbers, positive track IDs, and original video pixel coordinates. The annotations must describe the uploaded video. Model-based scoring uses those rows as evaluation ground truth. The synthetic demo reports tracking and recovery metrics from its annotation-driven run and requires no model weights.

Run history and artifacts are stored in `outputs/ui/<run_id>/` and persist across restarts. The server accepts up to four active or queued jobs, including evaluations, and processes one at a time. Playback uses an H.264 MP4 preview when encoding succeeds. The original annotated MP4 is retained in the run directory and served if conversion fails. Use `--output-dir` on `run_ui` to save runs elsewhere.

## Run the CPU quickstart

Run this command with the public ONNX model and sample clip already downloaded in this workspace. This model uses class 2 for drones. For a fresh checkout, the [synthetic tracking demo](#run-the-synthetic-tracking-demonstration) generates its own input files.

```bash
uv run --locked python -m scripts.run_video \
  --video outputs/onnx_smoke_source/video.mp4 \
  --config configs/default.yaml --model models/public-smoke-cerber-v2.onnx \
  --class-id 2 --output-dir outputs/quickstart
```

For your own footage, set `--video` to its path and `--model` to a local static YOLOv8 or YOLO11 ONNX export with raw class scores. Set `--input-size` to the export size and `--class-id` to the drone class in that model. The training workflow below creates `models/drone.onnx`. A single-class drone model uses class 0.

The quickstart uses a 45-frame, 1080p clip made by shifting one drone photograph. The expected result is one tracked box per frame with the same ID throughout. This exercises model inference, coordinate conversion, tracking, and file output on CPU. [Quickstart annotated video](outputs/quickstart/onnx_smoke_source_annotated.mp4)

ONNX weights and sample media are local files. In a fresh clone, obtain the public ONNX model below and supply your own video to `scripts.run_video` or the browser console.

```bash
mkdir -p models
curl -fL https://huggingface.co/MagistrTheOne/CERBER-CV-v2/resolve/main/detector_alpha_v2.onnx \
  -o models/public-smoke-cerber-v2.onnx
```

The tested public checkpoint has SHA-256 `702796ce6296d8ccb86f32f0c1b28bcc2074eb8a90f02108ab405db0dd7684c3`. [Public model card](https://huggingface.co/MagistrTheOne/CERBER-CV-v2)

To add **SkyGuard YOLO11n** in a fresh checkout, use the [training environment](#train-and-export) and export the publisher's verified `best.pt` checkpoint. The exporter verifies SHA-256, reads the checkpoint with PyTorch's weights-only loader, and transfers its tensors into a local YOLO11n architecture. The ONNX output uses drone class 0 and 640×640 inputs.

```bash
mkdir -p models
curl -fL https://huggingface.co/SkyGuardAI/drone-detection-yolov11/resolve/main/best.pt \
  -o models/skyguard-best.pt
.venv-train/bin/python -m scripts.export_skyguard models/skyguard-best.pt
```

The export was verified with Ultralytics 8.4.216 and PyTorch 2.15.0. [SkyGuard checkpoint](https://huggingface.co/SkyGuardAI/drone-detection-yolov11/tree/main)

## Test the implementation

```bash
uv run --locked pytest -q
```

The automated tests cover detector preprocessing and coordinate restoration, occlusion and re-entry, ambiguous matches, up to 15 simultaneous tracks, live backpressure, and evaluation metrics. Detector fixtures and Hailo mocks let the suite run on CPU without downloading weights.

## Run the synthetic tracking demonstration

In a fresh checkout, generate the source video and annotations once.

```bash
uv run --locked python -m scripts.make_demo outputs/synthetic_source
```

Run the tracker and evaluator on the generated fixture.

```bash
uv run --locked python -m scripts.run_video \
  --video outputs/synthetic_source/video.mp4 \
  --annotations outputs/synthetic_source/gt/gt.txt \
  --expected-frames 365 --output-dir outputs/synthetic
uv run --locked python -m scripts.evaluate \
  --ground-truth outputs/synthetic_source/gt/gt.txt \
  --predictions outputs/synthetic/synthetic_source.txt \
  --run-stats outputs/synthetic/synthetic_source_stats.json \
  --output outputs/synthetic/evaluation.json
```

The fixture contains two synthetic drones across 365 frames at 1920×1080 and 30 FPS. The expected result is the same global ID at frame 50, the last frame before the gap, and at frame 303, when the drone returns after 252 absent frames (8.4 seconds). Open the annotated video around frame 303 for the `Re-identified` label. The evaluation should show 478 box matches, one recovery with zero delay, zero ID switches, and IDF1 of 1.0. The overlay and statistics label this run as an oracle diagnostic.

## Where to find results

| Run | Annotated video | Prediction rows | Timing and evaluation |
| --- | --- | --- | --- |
| Browser test console | Play or download the overlay in Recent runs. | Download tracker predictions and raw detections from the run. | Inspect **Run statistics** and the optional **Evaluation** panel, or open the run folder under `outputs/ui/`. |
| CPU quickstart | [MP4](outputs/quickstart/onnx_smoke_source_annotated.mp4) | [TXT](outputs/quickstart/onnx_smoke_source.txt) | [Stats JSON](outputs/quickstart/onnx_smoke_source_stats.json) |
| Practice batch | Add `--render` to save an MP4 per sequence. | Open each sequence folder under the batch output directory. | Read the pooled `summary.json` and each sequence's `evaluation.json`. [Recorded results](docs/local_validation.md#organizer-practice-videos) |
| Synthetic tracking demo | [MP4](outputs/synthetic/synthetic_source_annotated.mp4) | [TXT](outputs/synthetic/synthetic_source.txt) | [Stats JSON](outputs/synthetic/synthetic_source_stats.json) and [evaluation JSON](outputs/synthetic/evaluation.json) |

- **Predictions.** Each headerless line contains `frame,id,left,top,width,height,confidence`. Frames start at 1, boxes use original image pixels, and empty frames have no rows. If you also pass `--detections PATH`, that file records the detector's original boxes before tracking with frame-local IDs. Its entries are limited to inference frames.
- **Stats.** Check `frames` and `detector_mode` first. `source_fps` is the video rate and `processing_fps` is the measured processing throughput. `timings_ms` breaks down detection, model inference where available, tracking, Re-ID, and rendering. `reidentified_events` counts recoveries reported by the tracker. Use the evaluation JSON to check those IDs against annotations.
- **Evaluation.** Check `tracking.idf1`, `tracking.id_switches`, and `recovery.details` for continuity and re-entry. Detector precision and recall require raw detections from a model run. Oracle evaluation uses the supplied run stats to identify the diagnostic and omit detector metrics.

For other videos, `--output-dir DIR` writes `<stem>.txt`, `<stem>_annotated.mp4`, and `<stem>_stats.json`. A file named `video.mp4` takes its stem from the parent folder. The generated outputs and local model weights are excluded from Git.

## Evaluate the organizer footage

The organizer practice set is under `datasets/practice` in this workspace. A fresh checkout needs the extracted `test_sample_clean.zip` from the Challenge 04 materials. Every sequence contains `video.mp4` and `gt/gt.txt`. The three clips decode to 511, 713, and 1,375 frames at 30 FPS. Ground truth uses 1-based frame numbers and pixel coordinates starting at zero.

```bash
uv run --locked python -m scripts.inspect_dataset practice datasets/practice --decode
uv run --locked python -m scripts.evaluate_dataset \
  --dataset datasets/practice --output-dir outputs/practice-evaluation \
  --model models/skyguard-yolo11n.onnx --class-id 0 --config configs/recall.yaml \
  --threads 4
```

The command writes one seven-column submission file per sequence inside its own subfolder, plus raw detections, timing and evaluation JSON, and a pooled `summary.json`. It prints detection F1, precision, recall, IDF1, ID switches, and MOTA from the submission tracks at IoU 0.50. Add `--render` to save annotated MP4s. Use a new or empty output directory each time.

Across the three practice videos, SkyGuard with Higher recall scored **87.3% detection F1**, **81.4% recall**, **54.2% IDF1**, and **75.2% MOTA**. CERBER with Balanced scored **82.9% detection F1**, **74.5% recall**, **56.5% IDF1**, and **68.5% MOTA**. To reproduce the balanced run, use `--model models/public-smoke-cerber-v2.onnx --class-id 2 --config configs/default.yaml`. [All six scores and per-run reports](docs/local_validation.md#organizer-practice-videos)

To score existing submissions, point `--pred-dir` at a folder of files named `<sequence>.txt` or at a previous batch results folder.

```bash
uv run --locked python -m scripts.evaluate_dataset \
  --dataset datasets/practice --pred-dir outputs/practice-evaluation \
  --output-dir outputs/practice-rescored
```

Use `--backend hailo --config configs/pi5_hailo.yaml --model models/drone.hef` for the Pi. The detector interval defaults to 1. Keep it at 1 when reporting raw detection F1, since raw detections are written only on inference frames. Evaluation assumes that truth and predictions use the same pixel origin. If the organizer defines 1-based positions, set the ground-truth offset to -1 for evaluation. Review the organizer README before submitting.

## Train and export

The training upload is available locally in `data/hackathon_dataset/`. It contains 16,833 train images and 5,919 validation images. The audit decoded all 22,752 images and matched 22,175 boxes, all with source class ID 1. It found 75 training images without a label file and 1,566 label files without a matching image. [Dataset inspection report](outputs/organizer_data/inspection.json)

Filename patterns put frames from the same recordings in both supplied splits. The pilot uses a separate copy with 512 training images and 106 validation images. It remaps drone class 1 to class 0, excludes training candidates from the validation sample's filename groups, and checks for byte-identical overlap. The manifest records every selected source file. Filename-based grouping is approximate. [Subset manifest](outputs/organizer_data/training_subset/manifest.json)

```bash
uv run --locked python -m scripts.inspect_dataset yolo data/hackathon_dataset --hash-images
```

For a fresh training environment, install the training dependencies once.

```bash
uv venv .venv-train --python 3.11
uv pip install --python .venv-train/bin/python -r requirements-train.txt
```

Train and export using the staged pilot dataset and the local pretrained YOLO11n checkpoint. Use a new run name for additional training and pass the checkpoint path printed by the trainer to the exporter.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-train/bin/python -m scripts.train \
  --data outputs/organizer_data/training_subset/dataset.yaml \
  --model models/yolo11n.pt --device mps --workers 0 \
  --imgsz 640 --epochs 4 --batch 8 --project outputs/organizer_data --name yolo11n_subset
.venv-train/bin/python -m scripts.export_model \
  --weights outputs/organizer_data/yolo11n_subset/weights/best.pt \
  --output models/drone.onnx --imgsz 640
```

The starter weights came from the official YOLO11 release. Use `--device cpu` or a CUDA device when MPS is unavailable. For a larger training run, group by recording before splitting and keep a separate capture-disjoint test set. The Pi guide covers HEF compilation with representative calibration footage. [YOLO11 release](https://github.com/ultralytics/assets/releases/tag/v8.3.0) and [Ultralytics export](https://docs.ultralytics.com/modes/export)

The four-epoch pilot is available at `models/drone.onnx` and uses class 0 at 640×640. On the same 106 validation images at IoU 0.50 and confidence 0.25, the public model scored **89.6% F1** and the pilot scored **64.4% F1**. The CPU quickstart uses the public model. The 106 images were also the pilot's validation set, so use a separate capture-disjoint test set to measure generalization. [Public detector report](outputs/organizer_data/public_baseline.json) and [pilot detector report](outputs/organizer_data/finetuned_baseline.json)

## Benchmark and improve

```bash
uv run --locked python -m scripts.benchmark --output outputs/quickstart/benchmark.json --repeat 2 -- \
  --video outputs/onnx_smoke_source/video.mp4 \
  --config configs/default.yaml --model models/public-smoke-cerber-v2.onnx \
  --class-id 2 --threads 4 --expected-frames 45
```

Open `outputs/quickstart/benchmark.json` and compare `median_processing_fps` and the individual runs. Detector timing includes preprocessing and postprocessing, and model timing measures backend forward calls. The report includes p50, p95, and p99 latency, process CPU usage, and peak memory. Live statistics include capture-to-output delay and application-level drops. Use a separate output path and pass `--render` before `--` to measure video encoding as part of the benchmark. That option uses a temporary MP4.

The recorded Mac runs on the sample clip measured about 16 FPS full-frame and 4.2 FPS with 2×2 tiling on every frame. The Balanced setting uses full-frame inference. Higher recall uses a conditional tile retry. Its accuracy was measured separately on the three practice videos. [Headless benchmark](outputs/onnx_smoke/benchmark_headless.json), [rendered benchmark](outputs/onnx_smoke/benchmark_render.json), [tiled benchmark](outputs/onnx_smoke/benchmark_tiles.json), and [practice scores](docs/local_validation.md#organizer-practice-videos)

1. Validate the detector on additional annotated footage. Examine misses by drone size and measure the cost of higher fixed input resolution.
2. Confirm the Hailo chip, compile the same trained weights, and benchmark the CPU and NPU pipelines on the Pi with the same video and settings. Record thermal and throttling state.
3. Review ID switches and failed recoveries by gap length. Long gaps, tiny or blurred crops, and visually similar drones can make identity ambiguous. Test any changed matching configuration on a separate validation set.

## Architecture reference

### Execution order by workflow

Each command below is a module under `scripts/`. For example, run `run_video` as `uv run --locked python -m scripts.run_video` from the project root. Use the workstation training environment for `train` and `export_model`.

| Workflow | Run in this order | Model and handoff |
| --- | --- | --- |
| Test a video in the browser | `run_ui → run_video → evaluate` when ground truth is supplied | The console queues an uploaded video with an available local ONNX detector. It serves the annotated video and saved outputs from `outputs/ui/<run_id>/`. The demo path uses the same synthetic fixture and annotation replay as the CLI workflow. |
| Try CPU inference | `run_video` | The 45-frame sample and `models/public-smoke-cerber-v2.onnx` go directly into the runner. Open the annotated MP4, predictions, and stats in `outputs/quickstart/`. |
| Test tracking and re-entry | `make_demo → run_video --annotations → evaluate` | The generator writes `video.mp4` and `gt/gt.txt`. The runner reads the boxes in oracle mode without loading model weights. The evaluator compares the resulting track IDs with the original ground truth. Open `outputs/synthetic/evaluation.json`. |
| Build a custom CPU model | `inspect_dataset yolo → train → export_model → run_video` | Start with YOLO images and labels, a dataset YAML, and a trusted local YOLOv8 or YOLO11 `.pt` checkpoint. Training writes `best.pt`. Export turns that checkpoint into `models/drone.onnx` for OpenCV. The README recipe uses YOLO11n. |
| Run on Raspberry Pi with Hailo | Confirm the chip and runtime → compile the trained model for that chip → `run_video` | Supply a compatible compiled HEF, such as `models/drone.hef`, and `configs/pi5_hailo.yaml`. Hailo runs the detector. The Pi CPU handles frame capture, tracking, re-identification, and writing results. [Pi 5 deployment guide](docs/raspberry_pi_setup.md) |
| Score the organizer videos | `inspect_dataset practice → evaluate_dataset` | The batch command invokes `run_video` and `evaluate` for each sequence using a compatible ONNX or HEF detector. It can also score saved submissions and writes sequence reports plus pooled metrics. |
| Measure performance | `benchmark → run_video` once per repeat | Pass the same model and video arguments after `--`. Benchmarking starts the runner in a fresh process each time and collects timing and host data in its own JSON report. |

### What each script does

| Script | Inputs and responsibility | Result |
| --- | --- | --- |
| [`run_ui.py`](scripts/run_ui.py) | Starts the loopback web server from the optional `ui` dependency group. `src/web/` serves the browser interface, handles uploads, and runs the existing inference and evaluation commands. | Local browser access to annotated video, run history, downloadable artifacts, and evaluation scores. |
| [`inspect_dataset.py`](scripts/inspect_dataset.py) | Inspects a dataset directory. `practice` reads video metadata and MOT annotations and reports box sizes and gaps. `--decode` counts actual frames. `yolo` checks image-to-label pairing, class IDs, and normalized coordinates. `--hash-images` checks exact duplicates across splits. | A JSON report on standard output. Use it before training or evaluating. |
| [`train.py`](scripts/train.py) | Fine-tunes a local Ultralytics detection checkpoint using the train and validation splits in a YOLO dataset YAML. Defaults are 640 pixels, 50 epochs, batch size 8, and seed 0. | Ultralytics training outputs under `outputs/train/` by default. The script prints the exact `best.pt` path. A first run typically writes `outputs/train/drone/weights/best.pt`. |
| [`export_model.py`](scripts/export_model.py) | Reads a standard detection `.pt` checkpoint and exports a static batch-one CPU ONNX model at the requested square size. The default is 640 pixels and opset 12. The graph exposes raw YOLO outputs for postprocessing in Python. | The `.onnx` file set by `--output`. HEF compilation uses the Hailo toolchain described in the deployment guide. |
| [`make_demo.py`](scripts/make_demo.py) | Creates a NumPy sky gradient and draws two drone shapes with OpenCV. Their positions follow fixed paths, including a gap at frames 51–302 for one drone. | A 365-frame MP4 and exact MOT ground truth in `gt/gt.txt`. Use a new destination when regenerating the fixture. |
| [`run_video.py`](scripts/run_video.py) | Reads the source, optional YAML, and CLI options. Constructs the selected detector and a fresh tracker, then runs the frame loop. | With `--output-dir`, writes predictions, an annotated MP4, and run stats. Add `--detections` to save boxes coming out of detector postprocessing. |
| [`evaluate.py`](scripts/evaluate.py) | Reads a single video's ground truth and predictions. Pass `--detections` for detector metrics and `--run-stats` for provenance and timing. | A JSON report with tracking scores, identity recovery, and detector scores when applicable. Use `--output` to save it to a file. |
| [`evaluate_dataset.py`](scripts/evaluate_dataset.py) | Runs and scores all annotated videos or scores existing submissions from `--pred-dir`. | Individual sequence reports, a terminal table with the six submission scores, and pooled `summary.json`. |
| [`export_skyguard.py`](scripts/export_skyguard.py) | Verifies and reads the SkyGuard checkpoint with a weights-only loader, rebuilds YOLO11n, and exports a raw static ONNX graph. | `models/skyguard-yolo11n.onnx` by default. Uses the separate training environment. |
| [`benchmark.py`](scripts/benchmark.py) | Repeats the chosen runtime configuration. The default is headless timing. `--render` adds overlays and encoding to a temporary MP4 per run. | JSON with median FPS, per-run timings, model and config hashes when provided, and available host telemetry. |

[`scripts/__init__.py`](scripts/__init__.py) marks the package that contains these commands.

### Runtime entry point and model selection

`run_video` accepts one source via `--video`, `--camera`, or `--stream`. Supply `--config` to load YAML. An explicit `--backend` takes precedence, then `--annotations` selects oracle mode, then the configured backend applies. Without those selections a `.hef` model path selects Hailo and everything else selects ONNX. `--model` overrides the model path in YAML. The presets leave the model path for you to supply. [Runner](scripts/run_video.py), [CPU preset](configs/default.yaml), and [Pi preset](configs/pi5_hailo.yaml)

| Backend | Accepted input | How it produces boxes |
| --- | --- | --- |
| ONNX CPU | A static batch-one raw YOLOv8 or YOLO11 detector export. The three local models use 640×640. CERBER-CV-v2 uses UAV class 2, and SkyGuard and the locally trained YOLO11n pilot use drone class 0. | OpenCV DNN resizes and pads each frame or tile, changes BGR to RGB, and scales the pixels to [0, 1]. It reads raw scores shaped `[1, 4 + class_count, anchors]`, selects the configured class, restores boxes to source coordinates, and applies non-max suppression. [ONNX backend](src/detectors/onnx.py), [CERBER-CV-v2 model card](https://huggingface.co/MagistrTheOne/CERBER-CV-v2), and [SkyGuard model card](https://huggingface.co/SkyGuardAI/drone-detection-yolov11) |
| Hailo | A HEF for the device's chip with a single RGB UINT8 image input and integrated per-class non-max suppression. The HEF provides the input dimensions. | HailoRT runs the graph on the accelerator. The backend reads the configured class's boxes and restores their source coordinates. [Hailo backend](src/detectors/hailo.py) |
| Oracle annotation replay | An offline video and MOT annotations supplied with `--annotations`. | The backend indexes annotations by source frame and supplies the annotated boxes to the tracker. It discards the ground-truth IDs before tracking. [Annotation backend](src/detectors/annotated.py) |

The CPU preset runs at 640×640 with a detector confidence cutoff of 0.10. The detector defaults are 0.45 IoU for non-max suppression and up to 100 retained boxes per inference frame. Normal detection starts with a full image. Higher recall retries with overlapping 2×2 crops when no full-frame box reaches 0.25 and merges the boxes in source coordinates. `--tiles 2` runs crops on every inference frame. `--detector-interval N` controls the spacing between inference frames. [Recall preset](configs/recall.yaml) and [shared detector geometry](src/detectors/base.py)

### Frame-to-output flow

1. **Capture and schedule.** OpenCV supplies the BGR image, a source frame index starting at 1, and an elapsed timestamp. An offline file is decoded in order and the declared FPS supplies its timestamps. Live capture uses a monotonic clock and keeps a one-frame buffer containing the newest frame. The run stats count gaps in the source indices reported by the application. OpenCV can also drop frames before the application sees them. Use `--max-frames` for a bounded live run or `--display` with a GUI-enabled OpenCV installation. [Capture](src/pipeline/capture.py) and [frame loop](src/pipeline/run.py)
2. **Detect and normalize coordinates.** All backends return `Detection` records with a box in original image pixels as `left, top, right, bottom`, a confidence, and a class ID. The frame loop can write these postprocessing results to the optional raw detection file before invoking the tracker. It passes an empty list when a scheduled detector call finds zero boxes and `None` when inference was intentionally skipped. [Shared records](src/types.py) and [frame loop](src/pipeline/run.py)
3. **Associate by motion.** A Kalman filter projects the center and dimensions of each local track. The tracker uses one-to-one assignment with overlap, center distance, and size similarity. It first matches detections at confidence 0.45 or higher in Balanced and 0.25 or higher in Higher recall. Detections at or above 0.10 and below the first-pass cutoff can extend an unmatched active track. A track becomes lost when a scheduled detection frame yields no match. It can resume via a strong motion match before the local state expires at 0.5 seconds since the last observation. The identity then becomes inactive. On intentionally skipped inference frames, active tracks can emit motion-predicted boxes with decaying confidence. [Motion filter](src/trackers/motion.py), [association](src/trackers/association.py), and [tracker](src/trackers/multi_object.py)
4. **Recover a global ID.** The tracker compares unmatched high-confidence boxes with up to 128 inactive identities retained for up to 30 seconds since last seen. Identity memory uses up to four 16×16 crop descriptors with color and gradient histograms, along with box size, motion, frame edges, and time since the last observation. The default threshold is 0.70 with a 0.06 margin over competing assignments where alternatives exist. A confident match reuses the global ID and creates a new local motion track. An ambiguous match creates a fresh ID. Re-ID uses frame pixels and in-memory state, and IDs start fresh each run. [Appearance features](src/reid/appearance.py) and [identity memory](src/reid/memory.py)
5. **Write predictions and timings.** The tracked output carries the global ID, box, confidence, and a re-entry flag. The writer clips coordinates to the image and serializes boxes as `left,top,width,height`. The optional overlay displays ID, confidence, recent trajectory, and `Re-identified` briefly after recovery. An offline annotated MP4 retains the input FPS. Requested files are staged and published after frame-count validation. The timing report includes per-stage means and p50, p95, and p99 in milliseconds, plus throughput, CPU use, and peak resident memory. [Writers](src/pipeline/output.py), [overlay](src/visualization/__init__.py), and [timings](src/pipeline/timing.py)

### Evaluation and benchmark boundaries

The evaluator uses one-to-one per-frame matches at IoU ≥ 0.5 by default. The team’s six metrics use the submitted tracks. ID switches compare a drone's current prediction ID with its last matched ID. IDF1 uses a global assignment of ground-truth and prediction identities across the video. MOTA is one minus the sum of false positives, false negatives, and ID switches divided by the number of ground-truth boxes. It can be negative. Batch scoring pools counts across sequences, with identity assignment performed separately for each video. Recovery reports delay and groups results by gap length. The raw detector section uses pre-tracker boxes and adds confidence calibration. Pass the run stats so annotation replay is labeled as an oracle diagnostic. [Team evaluation card](https://trello.com/c/Cjtp2XqM), [MOT reader](src/mot.py), and [metric implementation](src/evaluation/metrics.py)

The benchmark measures processing throughput from the runtime, after the model is loaded and the capture is opened. `source_fps` is the input rate. `processing_fps` is frames processed per elapsed second. `model_inference` times model forward calls, while `detector` also includes preprocessing and postprocessing. Use the benchmark's `--output` for the report and omit explicit runner output paths in repeated runs. [Benchmark runner](scripts/benchmark.py)

### Browser console and run storage

The static interface in `src/web/static/` submits uploads, polls run status, and displays the results returned by the API in `src/web/api.py`. One worker invokes `scripts.run_video` for inference, writes `preview.mp4` when encoding succeeds, and invokes `scripts.evaluate` when annotations are available. A run directory contains the input clip, `annotated.mp4`, `predictions.txt`, `stats.json`, and `detections.txt` for a model run. The browser reads the pipeline summary and its `timings_ms` fields from `stats.json` for **Run statistics**. Successful scoring adds `evaluation.json`, which populates the separate **Evaluation** panel. Attaching ground truth later uses the saved results. The CLI and browser both use the same tracking and evaluation code.

### Current verification and open work

- **CPU inference.** CERBER-CV-v2 produced one track across the 45-frame smoke clip made from one photograph. The organizer practice videos supplied the real-footage comparison of CERBER and SkyGuard.
- **Tracking.** The synthetic oracle run recovered the same ID on the first frame after the 252-frame gap, with zero ID switches and IDF1 of 1.0. It measures association with exact annotated boxes.
- **Training and export.** The four-epoch YOLO11n pilot fine-tuned on 512 uploaded images and exported to 640×640 ONNX. Both ONNX models ran on the same validation sample, with the public model scoring higher. [Training and validation logs](outputs/organizer_data/yolo11n_subset/results.csv)
- **Hailo.** The test suite exercises the backend with mocks. No HEF is present in this workspace. Hardware validation requires a compatible HEF and the physical Pi.
- **Organizer data.** The uploaded YOLO dataset enables still-image tests. The three labeled practice videos support video-level comparison and submission scoring.

[Recorded runs and report files](docs/local_validation.md)
