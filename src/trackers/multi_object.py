"""Causal two-stage motion tracking followed by persistent identity recovery."""

from dataclasses import dataclass
from math import exp, isfinite
from time import perf_counter
from typing import Mapping

import numpy as np

from src.reid.appearance import describe
from src.reid.memory import Identity, IdentityMemory, IdentityState
from src.types import Detection, Track

from .association import assign, motion_scores
from .motion import BoxFilter


@dataclass(slots=True)
class MotionTrack:
    local_id: int
    identity: Identity
    filter: BoxFilter


class MultiObjectTracker:
    """Maintain short-lived motion tracks and a separate global identity registry."""

    def __init__(
        self,
        high_confidence: float = 0.45,
        low_confidence: float = 0.10,
        max_lost_seconds: float = 0.5,
        reid_max_seconds: float = 30.0,
        reid_max_identities: int = 128,
        reid_min_similarity: float = 0.70,
        reid_ambiguity_margin: float = 0.06,
        reid_weights: Mapping[str, float] | None = None,
    ) -> None:
        if not 0 <= low_confidence <= high_confidence <= 1:
            raise ValueError("Tracker confidence thresholds must be ordered between 0 and 1")
        if not (isfinite(max_lost_seconds) and isfinite(reid_max_seconds)):
            raise ValueError("Tracker retention times must be finite")
        if max_lost_seconds <= 0 or reid_max_seconds < 0:
            raise ValueError("Local retention must be positive and Re-ID retention nonnegative")
        if not isinstance(reid_max_identities, int) or reid_max_identities < 0:
            raise ValueError("Dormant identity capacity must be a nonnegative integer")
        if not 0 <= reid_min_similarity <= 1 or not 0 <= reid_ambiguity_margin <= 1:
            raise ValueError("Re-ID similarity and margin must be between 0 and 1")

        self.high_confidence = high_confidence
        self.low_confidence = low_confidence
        self.max_lost_seconds = max_lost_seconds
        self.reid_min_similarity = reid_min_similarity
        self.reid_ambiguity_margin = reid_ambiguity_margin
        self.last_reid_seconds = 0.0
        self._motion_tracks: dict[int, MotionTrack] = {}
        self._next_local_id = 1
        self._identity_memory = IdentityMemory(reid_max_seconds, reid_max_identities, reid_weights)
        self._last_timestamp: float | None = None

    def update(
        self, detections: list[Detection] | None, frame: np.ndarray, timestamp: float, frame_index: int
    ) -> list[Track]:
        dt = 0.0 if self._last_timestamp is None else timestamp - self._last_timestamp
        if not isfinite(timestamp) or dt < 0:
            raise ValueError("Tracker timestamps must be finite and nondecreasing")
        self._last_timestamp = timestamp
        self.last_reid_seconds = 0.0
        height, width = frame.shape[:2]

        for track_id, track in list(self._motion_tracks.items()):
            if timestamp - track.identity.last_seen > self.max_lost_seconds:
                track.identity.state = IdentityState.INACTIVE
                del self._motion_tracks[track_id]
            elif dt:
                track.filter.predict(dt)
        start = perf_counter()
        self._identity_memory.expire(timestamp)
        self.last_reid_seconds += perf_counter() - start

        if detections is None:
            return self._predictions(timestamp, width, height)

        valid = []
        for detection in detections:
            if not self.low_confidence <= detection.confidence <= 1 or not all(map(isfinite, detection.box)):
                continue
            left, top, right, bottom = detection.box
            clipped = max(0.0, left), max(0.0, top), min(float(width), right), min(float(height), bottom)
            if clipped[0] < clipped[2] and clipped[1] < clipped[3]:
                valid.append(detection if clipped == detection.box else Detection(clipped, detection.confidence, detection.class_id))
        high = [detection for detection in valid if detection.confidence >= self.high_confidence]
        low = [detection for detection in valid if detection.confidence < self.high_confidence]
        tracks = list(self._motion_tracks.values())
        primary = self._associate(tracks, high, 0.36)
        observed_track_ids = {tracks[row].local_id for row, _ in primary}
        paired_high = {column for _, column in primary}
        eligible = [
            track for track in tracks
            if track.local_id not in observed_track_ids and track.identity.state is IdentityState.ACTIVE
        ]
        secondary = self._associate(eligible, low, 0.51)
        matched = [(tracks[row], high[column]) for row, column in primary]
        matched.extend((eligible[row], low[column]) for row, column in secondary)
        observed_track_ids.update(track.local_id for track, _ in matched)
        for track in tracks:
            track.identity.state = IdentityState.ACTIVE if track.local_id in observed_track_ids else IdentityState.LOST

        outputs = []
        for track, detection in matched:
            track.filter.correct(detection.box)
            vx, vy = (float(value) for value in track.filter.velocity[:2])
            track.identity.record(detection, frame_index, timestamp, (vx, vy))
            if detection.confidence >= self.high_confidence and timestamp - track.identity.appearance_at >= 0.30:
                start = perf_counter()
                track.identity.remember(describe(frame, detection.box), timestamp)
                self.last_reid_seconds += perf_counter() - start
            outputs.append(Track(track.identity.id, detection.box, detection.confidence))

        unmatched = [detection for index, detection in enumerate(high) if index not in paired_high]
        if unmatched:
            outputs.extend(self._recover(unmatched, frame, timestamp, frame_index, width, height))
        return sorted(outputs, key=lambda track: track.id)

    @staticmethod
    def _associate(
        tracks: list[MotionTrack], detections: list[Detection], minimum: float
    ) -> list[tuple[int, int]]:
        if not tracks or not detections:
            return []
        predicted = np.asarray([track.filter.box for track in tracks])
        observed = np.asarray([detection.box for detection in detections])
        uncertainty = np.sqrt([track.filter.variance[:2].sum() for track in tracks])
        return assign(motion_scores(predicted, observed, uncertainty), minimum)

    def _recover(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        timestamp: float,
        frame_index: int,
        width: int,
        height: int,
    ) -> list[Track]:
        start = perf_counter()
        candidates = [
            record for record in self._identity_memory.records.values() if record.state is IdentityState.INACTIVE
        ]
        appearances = [describe(frame, detection.box) for detection in detections]
        recovered: dict[int, Identity] = {}
        if candidates:
            scores = self._identity_memory.scores(candidates, detections, appearances, timestamp, (width, height))
            unambiguous = self._clear_matches(scores)
            recovered = {
                col: candidates[row]
                for row, col in assign(np.where(unambiguous, scores, -1), self.reid_min_similarity)
            }

        self.last_reid_seconds += perf_counter() - start
        outputs = []
        for col, (detection, features) in enumerate(zip(detections, appearances)):
            start = perf_counter()
            identity = recovered.get(col)
            reidentified = identity is not None
            if identity is None:
                identity = self._identity_memory.create(detection, frame_index, timestamp)
            else:
                identity.state = IdentityState.ACTIVE
                identity.record(detection, frame_index, timestamp, (0.0, 0.0))
            identity.remember(features, timestamp)
            self.last_reid_seconds += perf_counter() - start
            self._motion_tracks[self._next_local_id] = MotionTrack(self._next_local_id, identity, BoxFilter(detection.box))
            self._next_local_id += 1
            outputs.append(Track(identity.id, detection.box, detection.confidence, reidentified))
        return outputs

    def _clear_matches(self, scores: np.ndarray) -> np.ndarray:
        if self.reid_ambiguity_margin == 0:
            return np.ones(scores.shape, dtype=bool)
        sorted_rows = np.sort(scores, axis=1)
        sorted_columns = np.sort(scores, axis=0)
        row_runner_up = sorted_rows[:, -2, None] if scores.shape[1] > 1 else -1
        column_runner_up = sorted_columns[None, -2, :] if scores.shape[0] > 1 else -1
        return (
            (scores - row_runner_up >= self.reid_ambiguity_margin)
            & (scores - column_runner_up >= self.reid_ambiguity_margin)
        )

    def _predictions(self, timestamp: float, width: int, height: int) -> list[Track]:
        outputs = []
        for track in self._motion_tracks.values():
            if track.identity.state is not IdentityState.ACTIVE:
                continue
            left, top, right, bottom = track.filter.box
            box = max(0.0, left), max(0.0, top), min(float(width), right), min(float(height), bottom)
            if box[0] < box[2] and box[1] < box[3]:
                age = timestamp - track.identity.last_seen
                confidence = track.identity.observation.confidence * exp(-age / self.max_lost_seconds)
                outputs.append(Track(track.identity.id, box, confidence))
        return sorted(outputs, key=lambda track: track.id)
