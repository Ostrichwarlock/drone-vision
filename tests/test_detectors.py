"""Verify detector input and coordinate contracts without trained weights or a HAT."""

from contextlib import nullcontext
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src.detectors import AnnotatedDetector, HailoDetector, OnnxDetector


@pytest.fixture
def onnx_model(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    path.touch()
    net = Mock()
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda _: net)
    return path, net


def test_onnx_letterbox_class_filter_and_nms(onnx_model):
    path, net = onnx_model
    output = np.zeros((1, 6, 21), dtype=np.float32)
    output[0, :, :5] = np.array([
        [16, 16, 8, 4, 0.9, 0.2],
        [16, 16, 8, 4, 0.8, 0.2],
        [4, 16, 2, 2, 0.8, 0.9],
        [8, 4, 2, 2, 0.8, 0.1],
        [12, 18, 2, 2, 0.01, 0.1],
    ]).T
    net.forward.return_value = output
    detector = OnnxDetector(path, input_size=32, confidence=0.1)
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    frame[:, :, 2] = 255

    detections = detector.detect(frame, 1)

    assert len(detections) == 1
    assert detections[0].box == (24, 12, 40, 20)
    assert detections[0].confidence == pytest.approx(0.9)
    assert detector.last_inference_seconds >= 0
    blob = net.setInput.call_args.args[0]
    assert blob.shape == (1, 3, 32, 32)
    assert blob[0, :, 16, 16] == pytest.approx([1, 0, 0])
    assert blob[0, :, 0, 0] == pytest.approx([114 / 255] * 3)


def test_onnx_merges_overlap_in_original_coordinates(onnx_model):
    path, net = onnx_model
    outputs = np.zeros((2, 1, 5, 21), dtype=np.float32)
    outputs[0, 0, :, 0] = [16, 16, 4, 4, 0.7]
    outputs[1, 0, :, 0] = [8, 16, 4, 4, 0.9]
    net.forward.side_effect = outputs
    detector = OnnxDetector(path, input_size=32)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    detections = detector.detect(frame, 7, regions=[(0, 0, 64, 64), (16, 0, 80, 64)])

    assert len(detections) == 1
    assert detections[0].box == (28, 28, 36, 36)
    assert detections[0].confidence == pytest.approx(0.9)


def test_onnx_rejects_end_to_end_output(onnx_model):
    path, net = onnx_model
    net.forward.return_value = np.zeros((1, 300, 6), dtype=np.float32)
    detector = OnnxDetector(path, input_size=32)
    with pytest.raises(ValueError, match="raw"):
        detector.detect(np.zeros((32, 32, 3), dtype=np.uint8), 1)


def test_replay_exposes_only_current_boxes_and_no_gt_id_order(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text(
        "3,4,100,20,10,5,1,1,1\n"
        "1,1,30,5,10,5,1,1,1\n"
        "1,9,6,5,4,5,1,1,1\n"
        "1,8,50,5,4,5,0,0,0\n"
    )
    detector = AnnotatedDetector(path)
    frame = np.zeros((40, 120, 3), dtype=np.uint8)
    assert [d.box for d in detector.detect(frame, 1)] == [(6, 5, 10, 10), (30, 5, 40, 10)]
    assert detector.detect(frame, 2) == []
    assert len(detector.detect(frame, 3)) == 1


@pytest.fixture
def hailo_model(tmp_path, monkeypatch):
    path = tmp_path / "model.hef"
    path.touch()
    image = SimpleNamespace(shape=(32, 64, 3), format=SimpleNamespace(type="UINT8"))
    detections = SimpleNamespace(
        name="nms", format=SimpleNamespace(order="HAILO_NMS_BY_CLASS"),
        nms_shape=SimpleNamespace(number_of_classes=2),
    )
    hef = Mock()
    hef.get_network_group_names.return_value = ["yolo"]
    hef.get_input_vstream_infos.return_value = [image]
    hef.get_output_vstream_infos.return_value = [detections]
    model = Mock()
    model.output.return_value.shape = (66,)
    configured = Mock()
    model.configure.return_value = nullcontext(configured)
    physical = Mock()
    physical.control.identify.return_value.device_architecture = "HAILO8L"
    device = Mock()
    device.get_physical_devices.return_value = [nullcontext(physical)]
    device.create_infer_model.return_value = model
    vdevice = Mock(return_value=nullcontext(device))
    vdevice.create_params.return_value = SimpleNamespace(scheduling_algorithm=None)
    platform = SimpleNamespace(
        HEF=Mock(return_value=hef),
        HailoRTException=RuntimeError,
        HailoSchedulingAlgorithm=SimpleNamespace(ROUND_ROBIN=1),
        FormatType=SimpleNamespace(UINT8="UINT8", FLOAT32="FLOAT32"),
        FormatOrder=SimpleNamespace(NHWC="NHWC", HAILO_NMS_BY_CLASS="HAILO_NMS_BY_CLASS"),
        VDevice=vdevice,
    )
    monkeypatch.setitem(sys.modules, "hailo_platform", platform)
    return path, configured, detections


def test_hailo_runs_rgb_inference_and_decodes_per_class_nms(hailo_model):
    path, configured, _ = hailo_model
    bindings = configured.create_bindings.return_value
    bindings.output.return_value.get_buffer.return_value = [
        np.array([[0.3, 0.5, 0.4, 0.6, 0.99]]),
        np.array([[0.25, 0.25, 0.75, 0.75, 0.8], [0.4, 0.2, 0.5, 0.3, 0.1]]),
    ]
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    frame[:, :, 2] = 212
    with HailoDetector(path, class_id=1, expected_arch="hailo8l") as detector:
        results = detector.detect(frame, 1)
        assert detector.architecture == "hailo8l"
        assert detector.input_size == (64, 32)
        assert len(results) == 1
        assert results[0].box == (16, 8, 48, 24)
        assert results[0].class_id == 1
        assert results[0].confidence == pytest.approx(0.8)
        configured.run.assert_called_once_with([bindings], 10_000)
        buffer = bindings.input.return_value.set_buffer.call_args.args[0]
        assert buffer.dtype == np.uint8
        assert buffer[0, 0] == pytest.approx([212, 0, 0])
    with pytest.raises(RuntimeError, match="closed"):
        detector.detect(frame, 2)


def test_hailo_checks_output_schema_and_chip(hailo_model):
    path, _, info = hailo_model
    with pytest.raises(ValueError, match="detected hailo8l"):
        HailoDetector(path, expected_arch="hailo8")
    info.format.order = "NHWC"
    with pytest.raises(ValueError, match="integrated detection NMS"):
        HailoDetector(path)


def test_hailo_accepts_legacy_per_class_nms_order(hailo_model):
    path, configured, info = hailo_model
    platform = sys.modules["hailo_platform"]
    platform.FormatOrder = SimpleNamespace(NHWC="NHWC", HAILO_NMS="HAILO_NMS")
    info.format.order = "HAILO_NMS"
    bindings = configured.create_bindings.return_value
    bindings.output.return_value.get_buffer.return_value = [np.empty((0, 5)), np.empty((0, 5))]
    with HailoDetector(path) as detector:
        assert detector.detect(np.zeros((32, 64, 3), dtype=np.uint8), 1) == []
