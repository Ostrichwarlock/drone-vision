"""Per-frame detection, trajectory-based identity and disappearance metrics."""

from collections import Counter, defaultdict
from statistics import mean

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.mot import MotRow

Frames = dict[int, list[MotRow]]
RECOVERY_BUCKETS = ((1, 30), (31, 90), (91, 180), (181, None))
RECOVERY_OUTCOMES = ("recovered", "switched_identity", "missed_reappearance", "untracked_before_gap")


def box_ious(truth: list[MotRow], predictions: list[MotRow]) -> np.ndarray:
    a = np.asarray([row.box for row in truth], dtype=float).reshape(-1, 4)
    b = np.asarray([row.box for row in predictions], dtype=float).reshape(-1, 4)
    overlap = np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])
    intersections = np.prod(np.maximum(0.0, overlap), axis=2)
    unions = np.prod(a[:, 2:] - a[:, :2], axis=1)[:, None] + np.prod(b[:, 2:] - b[:, :2], axis=1) - intersections
    return np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)


def assign(ious: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    if not ious.size:
        return []
    valid = ious >= threshold
    priority = min(ious.shape) + 1
    # A valid pair takes priority over the total overlap of all assignments.
    scores = np.where(valid, priority + ious, 0)
    rows, columns = linear_sum_assignment(scores, maximize=True)
    return [(int(row), int(col)) for row, col in zip(rows, columns) if valid[row, col]]


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def object_counts(pairs: int, truth: int, detected: int) -> dict[str, float | int]:
    return {
        "true_positives": pairs,
        "false_positives": detected - pairs,
        "false_negatives": truth - pairs,
        "precision": rate(pairs, detected),
        "recall": rate(pairs, truth),
        "f1": rate(2 * pairs, detected + truth),
    }


def tracking_accuracy(truth_count: int, false_positives: int, false_negatives: int, id_switches: int) -> float | None:
    """Return CLEAR MOTA, which is undefined without ground-truth observations."""
    if truth_count == 0:
        return None
    return 1 - (false_positives + false_negatives + id_switches) / truth_count


def confidence_metrics(observations: list[tuple[float, bool]]) -> dict:
    bins = [[] for _ in range(10)]
    for confidence, matched in observations:
        bins[min(9, int(confidence * 10))].append((confidence, matched))
    reliability = []
    calibration_error = 0.0
    for index, rows in enumerate(bins):
        average_score = mean(confidence for confidence, _ in rows) if rows else None
        precision = mean(matched for _, matched in rows) if rows else None
        if rows:
            calibration_error += len(rows) * abs(average_score - precision)
        reliability.append({
            "min_inclusive": index / 10,
            "max": (index + 1) / 10,
            "count": len(rows),
            "mean_confidence": average_score,
            "precision": precision,
        })
    return {
        "brier_score": mean((confidence - matched) ** 2 for confidence, matched in observations) if observations else None,
        "expected_calibration_error": calibration_error / len(observations) if observations else None,
        "bins": reliability,
    }


def detector_metrics(truth: Frames, detections: Frames, threshold: float) -> dict:
    observations = []
    pair_count = 0
    for frame, detected in detections.items():
        pairs = assign(box_ious(truth.get(frame, []), detected), threshold)
        matched = {column for _, column in pairs}
        pair_count += len(pairs)
        observations.extend((row.confidence, index in matched) for index, row in enumerate(detected))
    return {
        **object_counts(pair_count, sum(map(len, truth.values())), len(observations)),
        "confidence_calibration": confidence_metrics(observations),
    }


def identity_metrics(
    potential_matches: Counter[tuple[int, int]],
    truth_count: int,
    prediction_count: int,
) -> dict[str, float | int]:
    # Identities with no overlap cannot contribute to IDTP and need no matrix row.
    truth_index = {identity: index for index, identity in enumerate(sorted({pair[0] for pair in potential_matches}))}
    prediction_index = {identity: index for index, identity in enumerate(sorted({pair[1] for pair in potential_matches}))}
    counts = np.zeros((len(truth_index), len(prediction_index)), dtype=np.int64)
    for (truth_id, prediction_id), count in potential_matches.items():
        counts[truth_index[truth_id], prediction_index[prediction_id]] = count
    rows, columns = linear_sum_assignment(counts, maximize=True)
    id_true_positives = int(counts[rows, columns].sum())
    return {
        "idtp": id_true_positives,
        "idfp": prediction_count - id_true_positives,
        "idfn": truth_count - id_true_positives,
        "id_precision": rate(id_true_positives, prediction_count),
        "id_recall": rate(id_true_positives, truth_count),
        "idf1": rate(2 * id_true_positives, prediction_count + truth_count),
    }


def track_pairs(
    truth: list[MotRow],
    predictions: list[MotRow],
    ious: np.ndarray,
    threshold: float,
    last_identity: dict[int, int],
) -> list[tuple[int, int]]:
    # Match the team's accumulator by keeping valid previous pairs before solving the rest.
    columns_by_id = {prediction.id: col for col, prediction in enumerate(predictions)}
    pairs = []
    used_columns = set()
    remaining_rows = []
    for row, item in enumerate(truth):
        col = columns_by_id.get(last_identity.get(item.id))
        if col is not None and col not in used_columns and ious[row, col] >= threshold:
            pairs.append((row, col))
            used_columns.add(col)
        else:
            remaining_rows.append(row)
    remaining_cols = [col for col in range(len(predictions)) if col not in used_columns]
    unmatched = ious[np.ix_(remaining_rows, remaining_cols)]
    pairs.extend((remaining_rows[row], remaining_cols[col]) for row, col in assign(unmatched, threshold))
    return pairs


def track_metrics(
    truth: Frames,
    predictions: Frames,
    threshold: float,
) -> tuple[dict, dict[int, list[tuple[int, int | None]]]]:
    potential_matches: Counter[tuple[int, int]] = Counter()
    mapped_observations: dict[int, list[tuple[int, int | None]]] = defaultdict(list)
    last_identity: dict[int, int] = {}
    identity_switches = 0
    pair_count = 0
    for frame in sorted(truth.keys() | predictions.keys()):
        current_truth = truth.get(frame, [])
        current_predictions = predictions.get(frame, [])
        ious = box_ious(current_truth, current_predictions)
        for row, column in zip(*np.nonzero(ious >= threshold)):
            potential_matches[current_truth[row].id, current_predictions[column].id] += 1

        pairs = track_pairs(current_truth, current_predictions, ious, threshold, last_identity)
        matches = {current_truth[row].id: current_predictions[col].id for row, col in pairs}
        pair_count += len(pairs)
        for row in current_truth:
            matching_id = matches.get(row.id)
            mapped_observations[row.id].append((frame, matching_id))
            if matching_id is not None:
                identity_switches += last_identity.get(row.id, matching_id) != matching_id
                last_identity[row.id] = matching_id

    truth_count = sum(map(len, truth.values()))
    prediction_count = sum(map(len, predictions.values()))
    return {
        **object_counts(pair_count, truth_count, prediction_count),
        **identity_metrics(potential_matches, truth_count, prediction_count),
        "id_switches": identity_switches,
        "mota": tracking_accuracy(
            truth_count, prediction_count - pair_count, truth_count - pair_count, identity_switches,
        ),
    }, mapped_observations


def observation_runs(observations: list[tuple[int, int | None]]) -> list[list[tuple[int, int | None]]]:
    runs = []
    for point in observations:
        if not runs or point[0] > runs[-1][-1][0] + 1:
            runs.append([])
        runs[-1].append(point)
    return runs


def recovery_summary(events: list[dict]) -> dict:
    counts = Counter(event["outcome"] for event in events)
    delays = [event["delay_frames"] for event in events if event["outcome"] == "recovered"]
    return {
        "events": len(events),
        **{outcome: counts[outcome] for outcome in RECOVERY_OUTCOMES},
        "failures": len(events) - counts["recovered"],
        "recovery_rate": counts["recovered"] / len(events) if events else None,
        "mean_delay_frames": mean(delays) if delays else None,
        "max_delay_frames": max(delays) if delays else None,
    }


def recovery_metrics(
    observations: dict[int, list[tuple[int, int | None]]],
    min_gap: int,
    window: int,
) -> dict:
    events = []
    for truth_id, matched in observations.items():
        runs = observation_runs(matched)
        for previous, current in zip(runs, runs[1:]):
            exit_frame, reentry_frame = previous[-1][0], current[0][0]
            gap = reentry_frame - exit_frame - 1
            if gap < min_gap:
                continue
            previous_id = next((identity for _, identity in reversed(previous) if identity is not None), None)
            current_frame, current_id = next(
                ((frame, identity) for frame, identity in current[:window] if identity is not None),
                (None, None),
            )
            if previous_id is None:
                outcome = "untracked_before_gap"
            elif current_id is None:
                outcome = "missed_reappearance"
            elif current_id != previous_id:
                outcome = "switched_identity"
            else:
                outcome = "recovered"
            events.append({
                "truth_id": truth_id,
                "last_visible_frame": exit_frame,
                "reentry_frame": reentry_frame,
                "gap_frames": gap,
                "previous_prediction_id": previous_id,
                "reappearance_prediction_id": current_id,
                "delay_frames": None if current_frame is None else current_frame - reentry_frame,
                "outcome": outcome,
            })
    events.sort(key=lambda event: (event["reentry_frame"], event["truth_id"]))
    return {
        **recovery_summary(events),
        "by_gap_frames": [
            {
                "min": lower,
                "max": upper,
                **recovery_summary([
                    event for event in events
                    if event["gap_frames"] >= lower and (upper is None or event["gap_frames"] <= upper)
                ]),
            }
            for lower, upper in RECOVERY_BUCKETS
        ],
        "details": events,
    }


def evaluate(
    truth: Frames,
    predictions: Frames,
    detections: Frames | None = None,
    *,
    iou_threshold: float = 0.5,
    min_gap_frames: int = 1,
    recovery_window_frames: int = 15,
) -> dict:
    if not 0 < iou_threshold <= 1 or min_gap_frames < 1 or recovery_window_frames < 1:
        raise ValueError("Use an IoU threshold in (0, 1] and positive frame counts")
    tracking, mapped_observations = track_metrics(truth, predictions, iou_threshold)
    return {
        "iou_threshold": iou_threshold,
        "min_gap_frames": min_gap_frames,
        "recovery_window_frames": recovery_window_frames,
        "detections": None if detections is None else detector_metrics(truth, detections, iou_threshold),
        "tracking": tracking,
        "recovery": recovery_metrics(mapped_observations, min_gap_frames, recovery_window_frames),
    }
