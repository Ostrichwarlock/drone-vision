"""Vectorized box association and assignment with an unmatched option."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(scores: np.ndarray, minimum: float) -> list[tuple[int, int]]:
    """Maximize valid matches first, then total similarity."""
    rows, columns = scores.shape
    if not rows or not columns:
        return []
    valid = scores >= minimum
    weights = np.zeros((rows, columns + rows))
    weights[:, :columns] = np.where(valid, min(rows, columns) + 1 + scores, -1)
    chosen_rows, chosen_cols = linear_sum_assignment(weights, maximize=True)
    return [
        (int(row), int(col))
        for row, col in zip(chosen_rows, chosen_cols)
        if col < columns and valid[row, col]
    ]


def motion_scores(predicted: np.ndarray, observed: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    sizes = predicted[:, 2:] - predicted[:, :2]
    observed_sizes = observed[:, 2:] - observed[:, :2]
    ratios = observed_sizes[None, :, :] / sizes[:, None, :]
    size_similarity = np.minimum(ratios, 1 / ratios).mean(axis=2)

    upper_left = np.maximum(predicted[:, None, :2], observed[None, :, :2])
    lower_right = np.minimum(predicted[:, None, 2:], observed[None, :, 2:])
    intersection = np.maximum(0.0, lower_right - upper_left).prod(axis=2)
    iou = intersection / (sizes.prod(axis=1)[:, None] + observed_sizes.prod(axis=1) - intersection)

    centers = (predicted[:, 2:] + predicted[:, :2]) / 2
    other_centers = (observed[:, 2:] + observed[:, :2]) / 2
    distance = np.linalg.norm(centers[:, None, :] - other_centers[None, :, :], axis=2)
    scale = np.linalg.norm(sizes, axis=1)
    radius = np.maximum(12.0, 0.7 * scale + np.minimum(2.5 * uncertainty, 1.5 * scale + 16))
    proximity = np.maximum(0.0, 1 - distance / radius[:, None])
    score = 0.45 * iou + 0.42 * proximity + 0.13 * size_similarity
    return np.where((distance <= radius[:, None]) & (size_similarity >= 0.45), score, -1.0)
