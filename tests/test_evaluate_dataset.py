import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from scripts.evaluate_dataset import aggregate_reports


def make_report(matches, false_positives, false_negatives, identity_matches, frames, elapsed):
    object_counts = {
        "true_positives": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
    return {
        "scope": "inference",
        "detections": object_counts,
        "tracking": {
            **object_counts,
            "idtp": identity_matches,
            "idfp": matches + false_positives - identity_matches,
            "idfn": matches + false_negatives - identity_matches,
            "id_switches": 1,
        },
        "recovery": {"details": []},
        "performance": {"frames": frames, "elapsed_seconds": elapsed},
    }


def test_dataset_totals_pool_counts_and_elapsed_time():
    report = aggregate_reports([
        make_report(8, 1, 0, 6, 4, 1.0),
        make_report(1, 0, 2, 1, 2, 2.0),
    ])
    assert report["detections"]["precision"] == pytest.approx(9 / 10)
    assert report["detections"]["recall"] == pytest.approx(9 / 11)
    assert report["detections"]["f1"] == pytest.approx(18 / 21)
    assert report["tracking"]["id_switches"] == 2
    assert report["tracking"]["idf1"] == pytest.approx(14 / 21)
    assert report["tracking"]["mota"] == pytest.approx(6 / 11)
    assert report["performance"]["processing_fps"] == 2
    assert report["recovery"]["recovery_rate"] is None


def test_batch_oracle_keeps_sequences_separate_and_applies_coordinate_offset(tmp_path: Path):
    dataset = tmp_path / "practice"
    for name in ("easy example", "hard example"):
        sequence = dataset / name
        sequence.mkdir(parents=True)
        video = cv2.VideoWriter(str(sequence / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (96, 96))
        assert video.isOpened()
        frame = np.zeros((96, 96, 3), dtype=np.uint8)
        for _ in range(4):
            video.write(frame)
        video.release()
        truth = sequence / "gt"
        truth.mkdir()
        (truth / "gt.txt").write_text("1,1,1,1,8,8,1,1,1\n2,1,1,1,8,8,1,1,1\n4,1,1,1,8,8,1,1,1\n")

    output = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.evaluate_dataset",
            "--dataset", str(dataset), "--output-dir", str(output),
            "--oracle", "--ground-truth-offset", "-1",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "summary.json").read_text())
    assert report["sequence_count"] == 2
    aggregate = report["aggregate"]
    assert aggregate["scope"] == "oracle_detection_diagnostic"
    assert aggregate["detections"] is None
    assert aggregate["tracking"]["idf1"] == 1
    assert aggregate["tracking"]["mota"] == 1
    assert aggregate["recovery"]["recovered"] == aggregate["recovery"]["events"] == 2
    assert aggregate["performance"]["frames"] == 8
    for sequence in report["sequences"]:
        assert (output / sequence["report"]).is_file()
        name = sequence["sequence"]
        assert (output / name / f"{name}.txt").read_text().splitlines()[0].split(",")[2:4] == ["0.000", "0.000"]


def test_flat_saved_predictions_print_team_metrics_without_video_or_timing(tmp_path: Path):
    dataset = tmp_path / "truth"
    predictions = tmp_path / "flat-predictions"
    predictions.mkdir()
    for name in ("easy", "hard"):
        truth = dataset / name / "gt"
        truth.mkdir(parents=True)
        (truth / "gt.txt").write_text("1,1,0,0,8,8,1,1,1\n")
    (predictions / "easy.txt").write_text("1,17,0,0,8,8,0.9\n")
    (predictions / "easy_raw.txt").write_text("1,0,30,30,8,8,0.9\n")
    (predictions / "hard.txt").write_text("")

    output = tmp_path / "reports"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.evaluate_dataset",
            "--dataset", str(dataset), "--pred-dir", str(predictions), "--output-dir", str(output),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert all(label in result.stdout for label in ("Detection F1", "Precision", "Recall", "IDF1", "ID switches", "MOTA", "OVERALL"))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["submission_metrics_source"] == "tracking"
    assert summary["aggregate"]["scope"] == "detector_provenance_unspecified"
    assert summary["aggregate"]["performance"] is None
    assert summary["aggregate"]["detections"] is None
    assert summary["aggregate"]["tracking"]["f1"] == pytest.approx(2 / 3)
    assert summary["aggregate"]["tracking"]["mota"] == 0.5
    assert summary["sequences"][0]["tracking"]["f1"] == 1
    assert summary["sequences"][0]["detections"]["f1"] == 0
