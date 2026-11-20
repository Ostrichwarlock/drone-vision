"""Portable detector backends."""

from src.detectors.annotated import AnnotatedDetector
from src.detectors.base import Detector
from src.detectors.hailo import HailoDetector
from src.detectors.onnx import OnnxDetector

__all__ = ["AnnotatedDetector", "Detector", "HailoDetector", "OnnxDetector"]
