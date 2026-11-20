"""Synthetic association tests with generated crops and controlled timestamps."""

import numpy as np
import pytest

from src.trackers import MultiObjectTracker
from src.types import Detection


RED = (10, 12, 220)
BLUE = (220, 12, 10)
GREEN = (12, 220, 10)


def observe(tracker, index, objects, timestamp=None):
    frame = np.zeros((240, 360, 3), dtype=np.uint8)
    detections = None
    if objects is not None:
        detections = []
        for box, color, confidence in objects:
            x1, y1, x2, y2 = box
            frame[y1:y2, x1:x2] = color
            detections.append(Detection(box, confidence))
    return tracker.update(detections, frame, index / 30 if timestamp is None else timestamp, index)


def drone(x, y, color=RED, confidence=0.95, size=16):
    return (x, y, x + size, y + size), color, confidence


def test_continuous_tracks_and_detector_skips_keep_ids():
    tracker = MultiObjectTracker()
    for index in range(1, 20):
        first, second = observe(tracker, index, [drone(30 + 3 * index, 40), drone(190, 130, BLUE)])
        assert (first.id, second.id) == (1, 2)
        assert first.box[0] == 30 + 3 * index
    first, second = observe(tracker, 20, None)
    assert (first.id, second.id) == (1, 2)
    assert first.box[0] == pytest.approx(90, abs=2)
    assert first.confidence < 0.95


def test_low_confidence_can_continue_only_an_existing_track():
    tracker = MultiObjectTracker()
    assert observe(tracker, 1, [drone(90, 70, confidence=0.24)]) == []
    first, = observe(tracker, 2, [drone(90, 70)])
    tracked, = observe(tracker, 3, [drone(91, 70, confidence=0.24), drone(180, 220, BLUE, 0.24)])
    assert first.id == tracked.id
    assert tracked.confidence == 0.24


def test_temporary_occlusion_does_not_emit_observed_or_ghost_boxes():
    tracker = MultiObjectTracker()
    for index in range(1, 7):
        first, = observe(tracker, index, [drone(50 + 2 * index, 60)])
    assert observe(tracker, 7, []) == []
    assert observe(tracker, 8, None) == []
    restored, = observe(tracker, 9, [drone(68, 60)])
    assert restored.id == first.id
    assert restored.reidentified is False


@pytest.mark.parametrize("age", [1.3, 8.4, 15])
def test_exit_and_reentry_restores_global_identity(age):
    tracker = MultiObjectTracker()
    for index in range(1, 7):
        first, = observe(tracker, index, [drone(index * 4 + 303, 85)])
    assert observe(tracker, 7, [], timestamp=1.1) == []
    returned, = observe(tracker, 8, [drone(333, 96)], timestamp=1.1 + age)
    assert returned.id == first.id
    assert returned.reidentified is True
    visible, = observe(tracker, 9, [drone(331, 96)], timestamp=1.1 + age + 1 / 30)
    assert visible.id == first.id
    assert visible.reidentified is False


def test_crossing_similar_drones_follow_motion():
    tracker = MultiObjectTracker()
    for index in range(1, 25):
        a, b = drone(30 + 4 * index, 90), drone(170 - 4 * index, 94)
        first, second = observe(tracker, index, sorted((a, b)))
        assert (first.box, second.box) == (a[0], b[0])


def test_full_overlap_with_one_detection_preserves_both_identities():
    tracker = MultiObjectTracker()
    for index in range(1, 22):
        a, b = drone(40 + 4 * index, 90), drone(120 - 4 * index, 90)
        objects = sorted((a, b)) if a != b else [a]
        output = observe(tracker, index, objects)
        if a == b:
            assert len(output) == 1
        else:
            assert {track.id: track.box for track in output} == {1: a[0], 2: b[0]}


def test_similar_drones_recover_on_distinct_compatible_edges():
    tracker = MultiObjectTracker()
    first, second = observe(tracker, 1, [drone(2, 65), drone(340, 65)])
    assert observe(tracker, 2, [], timestamp=2) == []
    recovered = observe(tracker, 3, [drone(340, 75), drone(3, 74)], timestamp=3)
    assert [track.id for track in recovered] == [first.id, second.id]
    assert [track.box[0] for track in recovered] == [3, 340]
    assert all(track.reidentified for track in recovered)


def test_ambiguous_return_creates_a_unique_id():
    tracker = MultiObjectTracker()
    observe(tracker, 1, [drone(125, 105), drone(195, 105)])
    assert observe(tracker, 2, [], timestamp=1.0) == []
    returned, = observe(tracker, 3, [drone(160, 105)], timestamp=1.1)
    assert returned.id == 3
    assert returned.reidentified is False


def test_one_dormant_id_is_never_assigned_twice():
    tracker = MultiObjectTracker(reid_ambiguity_margin=0)
    observe(tracker, 1, [drone(150, 95)])
    assert observe(tracker, 2, [], timestamp=1) == []
    outputs = observe(tracker, 3, [drone(139, 95), drone(161, 95)], timestamp=1.1)
    assert sorted(track.id for track in outputs) == [1, 2]
    assert sum(track.reidentified for track in outputs) == 1


def test_new_dissimilar_object_does_not_take_dormant_identity():
    tracker = MultiObjectTracker()
    observe(tracker, 1, [drone(335, 85)])
    assert observe(tracker, 2, [], timestamp=1.1) == []
    different, = observe(tracker, 3, [drone(335, 85, BLUE)], timestamp=1.2)
    assert different.id == 2
    assert different.reidentified is False


def test_tiny_distant_drones_can_continue_with_motion():
    tracker = MultiObjectTracker()
    for index in range(1, 15):
        track, = observe(tracker, index, [drone(23 + 3 * index, 36, size=3)])
        assert track.id == 1


def test_tiny_distant_drone_can_return_at_its_exit_edge():
    tracker = MultiObjectTracker()
    for index in range(1, 11):
        track, = observe(tracker, index, [drone(325 + 3 * index, 60, size=3)])
        assert track.id == 1
    observe(tracker, 11, [], timestamp=1)
    returned, = observe(tracker, 12, [drone(355, 61, size=3)], timestamp=9)
    assert (returned.id, returned.reidentified) == (1, True)


def test_partially_clipped_detections_keep_original_image_coordinates():
    tracker = MultiObjectTracker()
    frame = np.zeros((240, 360, 3), dtype=np.uint8)
    first, = tracker.update([Detection((350, 20, 367, 36), 0.95)], frame, 0.1, 1)
    second, = tracker.update([Detection((353, 21, 370, 37), 0.95)], frame, 0.13, 2)
    assert first.id == second.id == 1
    assert first.box == (350, 20, 360, 36)
    assert second.box == (353, 21, 360, 37)


def test_stale_and_excess_dormant_records_are_pruned():
    tracker = MultiObjectTracker(reid_max_seconds=4.0, reid_max_identities=1)
    observe(tracker, 1, [drone(2, 65), drone(340, 65, BLUE)])
    observe(tracker, 2, [], timestamp=1)
    new, recent = sorted(
        observe(tracker, 3, [drone(340, 65, BLUE), drone(2, 65)], timestamp=1.1),
        key=lambda track: track.box[0],
        reverse=True,
    )
    assert new.id == 3
    assert recent.id == 1
    observe(tracker, 4, [], timestamp=6)
    expired, = observe(tracker, 5, [drone(2, 65)], timestamp=6.1)
    assert expired.id == 4


def test_fifteen_simultaneous_objects_keep_unique_ids_with_reordered_input():
    tracker = MultiObjectTracker()
    expected = {}
    for index in range(1, 50):
        offset = 10 - abs(index % 20 - 10)
        objects = [
            drone(18 + (obj % 5) * 65 + offset, 30 + (obj // 5) * 70, GREEN, size=10)
            for obj in range(15)
        ]
        if index > 1:
            objects = objects[::-1] if index % 2 else objects[4:] + objects[:4]
        outputs = observe(tracker, index, objects)
        mapping = {(track.box[0] - offset, track.box[1]): track.id for track in outputs}
        if index == 1:
            expected = mapping
        assert len(mapping) == 15
        assert len(set(mapping.values())) == 15
        assert mapping == expected
