from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from inference.adapters.media import SampledFrame
from tests.conftest import SAMPLE_JPEG_BYTES, FakeInferenceEngine


def test_image_upload_returns_structured_detection_response(
    client: TestClient, fake_engine: FakeInferenceEngine
) -> None:
    response = client.post(
        "/v1/detect/image",
        files={"file": ("sample.jpg", SAMPLE_JPEG_BYTES, "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "fake-yolo"
    assert payload["image"] == {"width": 640, "height": 480}
    assert payload["detections"][0]["label"] == {"class_id": 0, "name": "person"}
    assert payload["detections"][0]["box"]["width"] == 10.0
    assert fake_engine.calls == [SAMPLE_JPEG_BYTES]


def test_rejects_invalid_image_content_type(client: TestClient) -> None:
    response = client.post(
        "/v1/detect/image",
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415


def test_rejects_empty_upload(client: TestClient) -> None:
    response = client.post(
        "/v1/detect/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400


def test_video_upload_returns_per_frame_detection_results(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_sample_video_bytes(
        video_bytes: bytes,
        sample_interval_seconds: float,
        max_frames: int,
    ) -> list[SampledFrame]:
        assert video_bytes == b"video bytes"
        assert sample_interval_seconds == 1.0
        assert max_frames == 60
        return [
            SampledFrame(frame_index=0, timestamp_seconds=0.0, image_bytes=b"frame-0"),
            SampledFrame(frame_index=30, timestamp_seconds=1.0, image_bytes=b"frame-1"),
        ]

    monkeypatch.setattr(
        "inference.entrypoints.routes.media.sample_video_bytes",
        fake_sample_video_bytes,
    )

    response = client.post(
        "/v1/detect/video",
        files={"file": ("sample.mp4", b"video bytes", "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sample_interval_seconds"] == 1.0
    assert [frame["frame_index"] for frame in payload["frames"]] == [0, 30]
    assert [frame["timestamp_seconds"] for frame in payload["frames"]] == [0.0, 1.0]


def test_webcam_unavailable_sends_error_and_closes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ClosedCapture:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    def open_closed_capture(_index: int | str) -> ClosedCapture:
        return ClosedCapture()

    monkeypatch.setattr(
        "inference.entrypoints.routes.media.open_capture",
        open_closed_capture,
    )

    with client.websocket_connect("/v1/detect/webcam") as websocket:
        assert websocket.receive_json()["error"] == "webcam_unavailable"


def test_webcam_streams_detection_payloads(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Capture:
        def __init__(self) -> None:
            self.frames = ["first", "second"]

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, str | None]:
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self) -> None:
            return None

    def open_stream_capture(_index: int | str) -> Capture:
        return Capture()

    def encode_frame(frame: str) -> bytes:
        return f"jpeg-{frame}".encode()

    monkeypatch.setattr(
        "inference.entrypoints.routes.media.open_capture",
        open_stream_capture,
    )
    monkeypatch.setattr(
        "inference.entrypoints.routes.media.encode_frame_to_jpeg",
        encode_frame,
    )

    with client.websocket_connect("/v1/detect/webcam") as websocket:
        first = websocket.receive_json()
        second = websocket.receive_json()

    assert first["frame_index"] == 0
    assert second["frame_index"] == 1
    assert first["model"] == "fake-yolo"
