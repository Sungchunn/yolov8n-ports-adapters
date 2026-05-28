from __future__ import annotations

import pytest

from inference.domain.exceptions import InvalidImageError
from inference.service_layer.services import detect_objects
from tests.conftest import FakeInferenceEngine


def test_detect_objects_delegates_to_inference_port() -> None:
    engine = FakeInferenceEngine()

    result = detect_objects(b"jpeg bytes", engine)

    assert result.model_name == "fake-yolo"
    assert engine.calls == [b"jpeg bytes"]


def test_detect_objects_rejects_empty_payload_before_engine_call() -> None:
    engine = FakeInferenceEngine()

    with pytest.raises(InvalidImageError):
        detect_objects(b"", engine)

    assert engine.calls == []
