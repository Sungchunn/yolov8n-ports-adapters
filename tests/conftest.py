from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from inference.config import Settings
from inference.domain.model import BoundingBox, Detection, DetectionLabel, InferenceResult
from inference.entrypoints.app import create_app
from inference.service_layer.ports import AbstractInferenceEngine


SAMPLE_JPEG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00"
    b"\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b"
    b"\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' "
    b"\",#\x1c\x1c(7),01444\x1f'9=82<.342"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x08"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xcf "
    b"\xff\xd9"
)


def _empty_call_log() -> list[bytes]:
    return []


@dataclass
class FakeInferenceEngine(AbstractInferenceEngine):
    calls: list[bytes] = field(default_factory=_empty_call_log)

    def predict(self, image_bytes: bytes) -> InferenceResult:
        self.calls.append(image_bytes)
        return InferenceResult(
            detections=(
                Detection(
                    box=BoundingBox(x1=1.0, y1=2.0, x2=11.0, y2=22.0),
                    label=DetectionLabel(class_id=0, name="person"),
                    confidence=0.9,
                ),
            ),
            image_width=640,
            image_height=480,
            inference_ms=12.5,
            model_name="fake-yolo",
        )


@pytest.fixture
def fake_engine() -> FakeInferenceEngine:
    return FakeInferenceEngine()


@pytest.fixture
def settings() -> Settings:
    return Settings(max_upload_bytes=1024, webcam_fps_limit=0)


@pytest.fixture
def client(fake_engine: FakeInferenceEngine, settings: Settings) -> Iterator[TestClient]:
    app = create_app(engine=fake_engine, settings=settings)
    with TestClient(app) as test_client:
        yield test_client
