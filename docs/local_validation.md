# Local validation

The development host was an ARM64 Mac. The project uses Python 3.11.18 with OpenCV 4.12.0, NumPy 2.2.6, and SciPy 1.16.3. uv resolved the runtime wheels for 64-bit Linux ARM on Python 3.11 and 3.13.

## Synthetic tracking diagnostic

The 365-frame 1080p fixture has two drone identities, with a 252-frame gap for one drone. The tracker recovered its original ID on frame 303, the first frame after the gap. It matched all 478 boxes, with zero ID switches and IDF1 of 1.0. The fixture uses ground-truth boxes as detection inputs, so its scores cover tracking and Re-ID. The two benchmark reports use three fresh Mac processes each.

- [Annotated MP4](../outputs/synthetic/synthetic_source_annotated.mp4)
- [Seven-column predictions](../outputs/synthetic/synthetic_source.txt)
- [Evaluation and timings](../outputs/synthetic/evaluation.json)
- [Headless benchmark](../outputs/synthetic/benchmark_headless.json) and [rendered benchmark](../outputs/synthetic/benchmark_render.json)

## CPU inference smoke test

OpenCV DNN ran the public CERBER-CV-v2 YOLOv8 model at 640 pixels, using UAV class 2. The 45-frame, 1080p test video was made from one public drone image with small translations. The program produced one box and the same track ID on each frame. This tests actual inference, geometry, and output. The test supplies no accuracy estimate for independent footage. [Model by MagistrTheOne](https://huggingface.co/MagistrTheOne/CERBER-CV-v2), [source image by AXERA-TECH](https://huggingface.co/AXERA-TECH/Drone-axera/blob/main/test/23.jpg)

| 1080p smoke test on Mac, 4 OpenCV threads | Runs | Median processing FPS |
| --- | ---: | ---: |
| Full frame, headless | 2 | 15.64 |
| Full frame, annotated video | 2 | 15.97 |
| 2×2 tiles, headless | 1 | 4.22 |

The full-frame runs have similar throughput at this sample size. Running tiles on every frame substantially increased inference cost. Higher recall instead retries with tiles only when the full-frame result lacks a box at confidence 0.25.

- [Annotated MP4](../outputs/onnx_smoke/onnx_smoke_source_annotated.mp4) and [seven-column predictions](../outputs/onnx_smoke/onnx_smoke_source.txt)
- [Headless benchmark](../outputs/onnx_smoke/benchmark_headless.json), [rendered benchmark](../outputs/onnx_smoke/benchmark_render.json), and [tiled benchmark](../outputs/onnx_smoke/benchmark_tiles.json)

The benchmark reports include model and configuration hashes, invocation arguments, timing percentiles, process CPU, and peak RSS. A single fully utilized core corresponds to 100% process CPU.

## Training and export verification

In the separate uv training environment, Ultralytics 8.4.216 and PyTorch 2.15.0 completed one CPU epoch of YOLO11n on 18 images sampled from the synthetic fixture. The provided export command produced a static 160×160 ONNX graph, which OpenCV loaded and ran for five frames. This verifies the training, export, and model-loading commands. The single synthetic epoch yielded zero validation mAP and is unsuitable for deployment.

- [Training configuration](../outputs/training_smoke/run/args.yaml) and [training results](../outputs/training_smoke/run/results.csv)
- [Exported model inference statistics](../outputs/training_smoke/inference_stats.json)

## Organizer practice videos

The organizer's `test_sample_clean.zip` contains the easy, medium, and hard annotated practice clips. The local copy is under `datasets/practice/`. All three clips are 1920×1080 at 30 FPS. Together they contain 2,599 frames and 3,595 ground-truth boxes. The archive uses 1-based frame numbers and zero-origin pixel coordinates. Its source is the organizer's Challenge 04 materials. [Team evaluation card](https://trello.com/c/Cjtp2XqM), [dataset inspection](../outputs/practice_inspection.json)

The CPU runs process all frames with 640×640 inputs. The scores below use the submitted tracking rows at IoU 0.50 and pool the three sequences. The balanced run matched the team's Trello evaluator on all six metrics.

| Model and profile | Detection F1 | Precision | Recall | IDF1 | ID switches | MOTA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CERBER-CV-v2, Balanced | 82.94% | 93.60% | 74.46% | 56.51% | 33 | 68.46% |
| CERBER-CV-v2, Higher recall | 84.22% | 89.46% | 79.55% | 48.56% | 52 | 68.73% |
| SkyGuard YOLO11n, Higher recall | 87.27% | 94.05% | 81.39% | 54.16% | 38 | 75.19% |

SkyGuard with Higher recall improved hard-clip detection F1 from 75.92% to 83.17% compared with CERBER Balanced. It recovered more labeled boxes overall and produced five additional ID switches. CERBER Balanced had the best pooled IDF1. Raw detector F1 was 82.59% for CERBER Balanced, 81.75% for CERBER Higher recall, and 85.85% for SkyGuard Higher recall. The runs shared the development machine during parts of the comparison, so their processing FPS are unsuitable for model throughput comparisons.

- [CERBER Balanced report](../outputs/practice-default-scored/summary.json)
- [CERBER Higher recall report](../outputs/practice-recall/summary.json)
- [SkyGuard Higher recall report](../outputs/practice-skyguard/summary.json)

On the fixed 106-image validation sample below, conditional tiles with CERBER raised raw recall at confidence 0.25 from 86% to 93% and F1 from 89.6% to 92.5%. Of 22 tiny labeled objects, detections increased from 17 to 21. This retried 18 of the 106 images. The retry is triggered by the frame's highest-scoring box, so an already-confident detection can prevent a retry for another object in the frame. [Still-image profile comparison](../outputs/organizer_data/public_recognition_comparison.json)

## Remaining hardware validation

The Hailo backend passed mocked binding, preprocessing, NMS decoding, and architecture checks. It was reviewed against HailoRT 4.x and 5.x Python bindings. An actual Pi run requires the chip identification, its HailoRT version, and a compatible drone HEF. The project has no measured Pi FPS, temperature, or accelerator utilization. [HailoRT 4.19 bindings](https://github.com/hailo-ai/hailort/blob/v4.19.0/hailort/libhailort/bindings/python/platform/hailo_platform/pyhailort/pyhailort.py), [HailoRT 5.1 bindings](https://github.com/hailo-ai/hailort/blob/v5.1.1/hailort/libhailort/bindings/python/platform/hailo_platform/pyhailort/pyhailort.py)

## Uploaded YOLO images

The dataset audit decoded 22,752 images and found 22,175 valid boxes, all labeled class 1. There are 75 training images with missing labels and 1,566 label files without paired images. No byte-identical images cross the supplied train and validation splits. Adjacent frames from timestamped captures do cross those splits. [Dataset audit](../outputs/organizer_data/inspection.json)

For a first training run, 512 images were copied to a separate dataset and the class IDs were remapped to 0. Selection excluded training candidates in the 106-image validation sample's filename groups and any byte-identical matches. The validation set has 100 positive images and six explicit negatives. Filename grouping is a capture heuristic. The validation images also informed the checkpoint selection. [Selection manifest](../outputs/organizer_data/training_subset/manifest.json) and [staged dataset audit](../outputs/organizer_data/subset_inspection.json)

YOLO11n fine-tuned for four epochs at 640×640 on Mac MPS and exported to `models/drone.onnx`. Both ONNX models were scored with OpenCV on the same 106 images at IoU 0.50 and confidence 0.25. The public model's original training overlap with the sample is unknown. [Training curves CSV](../outputs/organizer_data/yolo11n_subset/results.csv) and [checkpoint](../outputs/organizer_data/yolo11n_subset/weights/best.pt)

| Model | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Public CERBER-CV-v2 | 93.5% | 86.0% | 89.6% |
| Four-epoch YOLO11n pilot | 74.0% | 57.0% | 64.4% |

Both reports include confidence thresholds 0.10, 0.25, and 0.45, and they record the sample paths and per-image counts. [Public report](../outputs/organizer_data/public_baseline.json), [pilot report](../outputs/organizer_data/finetuned_baseline.json), and [local comparison script](../outputs/organizer_data/score_onnx_sample.py)
