from __future__ import annotations

import pytest

from inference.domain.exceptions import InvalidBoundingBox, InvalidDetection
from inference.domain.model import BoundingBox, Detection, DetectionLabel, InferenceResult


def test_bounding_box_exposes_derived_geometry() -> None:
    box = BoundingBox(x1=10, y1=20, x2=30, y2=50)

    assert box.width == 20
    assert box.height == 30
    assert box.area == 600
    assert box.center == (20, 35)


def test_bounding_box_rejects_negative_coordinates() -> None:
    with pytest.raises(InvalidBoundingBox):
        BoundingBox(x1=-1, y1=0, x2=1, y2=1)


def test_detection_rejects_confidence_outside_probability_range() -> None:
    with pytest.raises(InvalidDetection):
        Detection(
            box=BoundingBox(x1=0, y1=0, x2=1, y2=1),
            label=DetectionLabel(class_id=1, name="car"),
            confidence=1.1,
        )


def test_inference_result_normalizes_detections_to_tuple() -> None:
    result = InferenceResult(
        detections=[],
        image_width=1,
        image_height=1,
        inference_ms=0,
        model_name="fake",
    )

    assert result.detections == ()
    assert result.count() == 0
