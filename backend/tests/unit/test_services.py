from __future__ import annotations

import pytest

from inference.domain.model import MediaKind
from inference.service_layer.errors import InvalidImageError
from inference.service_layer.media import MediaFormat
from inference.service_layer.services import detect_media, detect_objects
from tests.support import FakeInferenceEngine, FakeMediaProcessor


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


def test_detect_media_processes_image_before_inference() -> None:
    engine = FakeInferenceEngine()
    media_processor = FakeMediaProcessor()

    result = detect_media(b"jpeg bytes", MediaFormat.JPEG, engine, media_processor)

    assert result.kind == MediaKind.IMAGE
    assert len(result.frames) == 1
    assert engine.calls == [b"jpeg bytes"]
    assert media_processor.calls == [(b"jpeg bytes", MediaFormat.JPEG)]


def test_detect_media_processes_each_video_frame() -> None:
    engine = FakeInferenceEngine()
    media_processor = FakeMediaProcessor()

    result = detect_media(b"video bytes", MediaFormat.AVI, engine, media_processor)

    assert result.kind == MediaKind.VIDEO
    assert result.sample_interval_seconds == 0.25
    assert [frame.frame_index for frame in result.frames] == [0, 8]
    assert engine.calls == [b"frame-0", b"frame-1"]


def test_detect_media_rejects_empty_payload_before_media_processor_call() -> None:
    engine = FakeInferenceEngine()
    media_processor = FakeMediaProcessor()

    with pytest.raises(InvalidImageError):
        detect_media(b"", MediaFormat.JPEG, engine, media_processor)

    assert engine.calls == []
    assert media_processor.calls == []
