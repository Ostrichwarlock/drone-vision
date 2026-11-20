import json
import sys

import numpy as np
import pytest

from scripts import evaluate as evaluate_cli
from scripts.benchmark import summarize_telemetry, telemetry_sample
from src.evaluation import evaluate
from src.evaluation.metrics import assign
from src.mot import MotRow, read_mot


BOX_A = (0, 0, 10, 10)
BOX_B = (30, 30, 38, 38)


def test_detector_counts_ignore_track_ids_and_calibration_scores_matches():
    truth = {1: [MotRow(2, BOX_A, 1)], 2: [MotRow(2, BOX_A, 1)]}
    raw = {1: [MotRow(-1, BOX_A, 0.9), MotRow(-1, BOX_B, 0.2)]}
    result = evaluate(truth, {}, raw)
    detections = result["detections"]
    assert detections["true_positives"] == 1
    assert detections["false_positives"] == 1
    assert detections["false_negatives"] == 1
    assert detections["precision"] == detections["recall"] == detections["f1"] == 0.5
    assert detections["confidence_calibration"]["brier_score"] == pytest.approx(0.025)
    assert detections["confidence_calibration"]["expected_calibration_error"] == pytest.approx(0.15)
    assert result["tracking"]["idf1"] == 0


def test_frame_assignment_maximizes_cardinality_before_overlap():
    ious = np.array([[1.00, 0.30, 0.00], [0.00, 1.00, 0.30], [0.30, 0.00, 0.00]])
    assert assign(ious, 0.25) == [(0, 1), (1, 2), (2, 0)]


def test_global_idf1_and_switches_after_a_gap():
    truth = {i: [MotRow(1, BOX_A, 1)] for i in (1, 2, 200, 201, 202)}
    predicted = {i: [MotRow(7 if i < 200 else 8, BOX_A, 1)] for i in truth}
    result = evaluate(truth, predicted)
    tracking = result["tracking"]
    assert tracking["f1"] == 1
    assert tracking["id_switches"] == 1
    assert tracking["idtp"] == 3
    assert tracking["idf1"] == 0.6
    assert tracking["mota"] == 0.8
    recovery = result["recovery"]
    assert recovery["recovered"] == 0
    assert recovery["failures"] == 1
    assert recovery["by_gap_frames"][-1]["events"] == 1
    assert recovery["details"][0]["gap_frames"] == 197
    assert recovery["details"][0]["outcome"] == "switched_identity"


def test_submission_tracking_keeps_valid_previous_pair_after_gap():
    truth = {1: [MotRow(1, BOX_A, 1)], 3: [MotRow(1, BOX_A, 1)]}
    predicted = {
        1: [MotRow(9, BOX_A, 1)],
        3: [MotRow(8, BOX_A, 1), MotRow(9, (1, 0, 11, 10), 1)],
    }
    result = evaluate(truth, predicted)
    assert result["tracking"]["id_switches"] == 0
    assert result["tracking"]["mota"] == 0.5
    assert result["recovery"]["recovered"] == 1


def test_mota_can_be_negative_and_is_undefined_without_truth():
    predictions = {1: [MotRow(2, BOX_B, 1)], 2: [MotRow(2, BOX_B, 1)]}
    assert evaluate({1: [MotRow(1, BOX_A, 1)]}, predictions)["tracking"]["mota"] == -2
    assert evaluate({}, predictions)["tracking"]["mota"] is None


def test_recovery_window_reports_delay_and_misses():
    truth = {i: [MotRow(41, BOX_A, 1)] for i in (1, 2, 5, 6, 7)}
    predicted = {i: [MotRow(62, BOX_A, 1)] for i in (1, 2, 6, 7)}
    result = evaluate(truth, predicted)
    assert result["tracking"]["idf1"] == pytest.approx(8 / 9)
    assert result["tracking"]["id_switches"] == 0
    assert result["recovery"]["mean_delay_frames"] == 1
    assert result["recovery"]["recovery_rate"] == 1
    assert evaluate(truth, predicted, recovery_window_frames=1)["recovery"]["missed_reappearance"] == 1
    assert evaluate(truth, predicted, min_gap_frames=3)["recovery"]["events"] == 0
    assert evaluate(truth, {6: predicted[6]})["recovery"]["untracked_before_gap"] == 1


def test_mot_shared_parser_coordinates_and_duplicates(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("2,1,1,2,8,12,1,1,1\n3,1,0,0,8,12,0,1,1\n")
    assert read_mot(path, ground_truth=True, coordinate_offset=-1) == {2: [MotRow(1, (0, 1, 8, 13), 1)]}
    path.write_text("1,1,0,0,8,8,0.5\n1,1,0,0,8,8,0.5\n")
    with pytest.raises(ValueError, match="duplicate ID"):
        read_mot(path)


def test_oracle_report_labels_tracking_and_excludes_detector_scores(tmp_path, monkeypatch, capsys):
    truth = tmp_path / "gt.txt"
    truth.write_text("1,1,0,0,8,8,1,1,1\n")
    predictions = tmp_path / "predictions.txt"
    predictions.write_text("1,17,0,0,8,8,0.9\n")
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"detector_mode": "oracle", "processing_fps": 31.4}))
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--ground-truth", str(truth), "--predictions", str(predictions),
        "--detections", str(predictions), "--run-stats", str(stats),
    ])
    assert evaluate_cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["detections"] is None
    assert report["scope"] == "oracle_detection_diagnostic"
    assert report["tracking"]["idf1"] == 1
    assert report["performance"]["processing_fps"] == 31.4


def test_pi_throttle_reports_current_and_historical_flags(tmp_path, monkeypatch):
    sensor = tmp_path / "temp"
    sensor.write_text("50123\n")
    monkeypatch.setattr("scripts.benchmark.firmware_value", lambda *_: "throttled=0x40004")
    summarized = summarize_telemetry([telemetry_sample(sensor, "mock", 0.0)])
    assert summarized["cpu_temperature_peak_c"] == 50.123
    assert summarized["throttling_flags_observed"] == ["throttled", "throttling_has_occurred"]
