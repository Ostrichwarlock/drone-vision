"""Constant-velocity Kalman filter for center and box size."""

import numpy as np

from src.types import Box


def measurement(box: Box) -> np.ndarray:
    left, top, right, bottom = box
    return np.array([(left + right) / 2, (top + bottom) / 2, right - left, bottom - top])


class BoxFilter:
    def __init__(self, box: Box) -> None:
        self.position = measurement(box)
        self.velocity = np.zeros(4)
        self.variance = self._measurement_variance()
        self.covariance = np.zeros(4)
        self.velocity_variance = (self._scale() * 16) ** 2

    def _scale(self) -> np.ndarray:
        width, height = np.maximum(self.position[2:], 4.0)
        return np.array([width, height, width, height])

    def _measurement_variance(self) -> np.ndarray:
        return np.maximum(1.0, self._scale() * 0.1) ** 2

    @property
    def box(self) -> Box:
        x, y, width, height = (float(value) for value in self.position)
        return x - width / 2, y - height / 2, x + width / 2, y + height / 2

    def predict(self, dt: float) -> None:
        acceleration_variance = (self._scale() * 35) ** 2
        self.position += self.velocity * dt
        self.position[2:] = np.maximum(self.position[2:], 1.0)
        self.variance += 2 * dt * self.covariance + dt ** 2 * self.velocity_variance
        self.variance += acceleration_variance * dt ** 4 / 4
        self.covariance += dt * self.velocity_variance + acceleration_variance * dt ** 3 / 2
        self.velocity_variance += acceleration_variance * dt ** 2

    def correct(self, box: Box) -> None:
        residual = measurement(box) - self.position
        innovation_variance = self.variance + self._measurement_variance()
        position_gain = self.variance / innovation_variance
        velocity_gain = self.covariance / innovation_variance
        self.position += position_gain * residual
        self.velocity += velocity_gain * residual
        self.velocity_variance -= velocity_gain * self.covariance
        self.covariance *= 1 - position_gain
        self.variance *= 1 - position_gain
