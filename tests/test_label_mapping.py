import pytest

from pipeline.ingest.common import map_fall_label, map_har_label, map_label
from pipeline.schema import TASK_FALL, TASK_HAR


def test_known_har_label_mappings():
    assert map_har_label("walking") == "locomotion"
    assert map_har_label("WALKING_UPSTAIRS") == "stairs"
    assert map_har_label("laying") == "static"
    assert map_har_label("ironing") == "other"


def test_known_fall_label_mappings():
    assert map_fall_label("fall") == "fall"
    assert map_fall_label("FALL_FORWARD") == "fall"
    assert map_fall_label("ADL") == "non_fall"
    assert map_fall_label("walking") == "non_fall"


def test_unknown_har_label_behavior_default_and_strict():
    assert map_har_label("unseen_label") == "other"
    with pytest.raises(ValueError):
        map_har_label("unseen_label", unknown_strategy="raise")


def test_unknown_fall_label_behavior_default_and_strict():
    assert map_fall_label("mystery_event") == "non_fall"
    with pytest.raises(ValueError):
        map_fall_label("mystery_event", unknown_strategy="raise")


def test_map_label_dispatch_by_task_type():
    assert map_label("walking", TASK_HAR) == "locomotion"
    assert map_label("fall", TASK_FALL) == "fall"
    with pytest.raises(ValueError):
        map_label("walking", "invalid_task")
