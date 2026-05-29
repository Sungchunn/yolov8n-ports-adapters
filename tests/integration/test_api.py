from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import (
    SAMPLE_JPEG_BYTES,
    SAMPLE_PNG_BYTES,
    FakeInferenceEngine,
    FakeMediaProcessor,
)


def test_frontend_index_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Vision Inference" in response.text


def test_asset_manifest_is_served(client: TestClient) -> None:
    response = client.get("/assets/video/videos.json")

    assert response.status_code == 200
    assert response.json()[0]["contentType"] == "video/x-msvideo"


def test_upload_jpeg_returns_image_detection_response(
    client: TestClient,
    fake_engine: FakeInferenceEngine,
    fake_media_processor: FakeMediaProcessor,
) -> None:
    response = client.post(
        "/v1/detect/upload",
        files={"file": ("sample.jpg", SAMPLE_JPEG_BYTES, "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "image"
    assert payload["result"]["model"] == "fake-yolo"
    assert payload["result"]["image"] == {"width": 640, "height": 480}
    assert payload["result"]["detections"][0]["label"] == {
        "class_id": 0,
        "name": "person",
    }
    assert fake_engine.calls == [SAMPLE_JPEG_BYTES]
    assert fake_media_processor.calls == [(SAMPLE_JPEG_BYTES, "image/jpeg")]


def test_upload_png_returns_image_detection_response(client: TestClient) -> None:
    response = client.post(
        "/v1/detect/upload",
        files={"file": ("sample.png", SAMPLE_PNG_BYTES, "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["kind"] == "image"


def test_upload_avi_returns_per_frame_detection_results(
    client: TestClient,
    fake_engine: FakeInferenceEngine,
    fake_media_processor: FakeMediaProcessor,
) -> None:
    response = client.post(
        "/v1/detect/upload",
        files={"file": ("sample.avi", b"video bytes", "video/x-msvideo")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "video"
    assert payload["sample_interval_seconds"] == 1.0
    assert [frame["frame_index"] for frame in payload["frames"]] == [0, 30]
    assert [frame["timestamp_seconds"] for frame in payload["frames"]] == [0.0, 1.0]
    assert fake_engine.calls == [b"frame-0", b"frame-1"]
    assert fake_media_processor.calls == [(b"video bytes", "video/x-msvideo")]


def test_rejects_invalid_upload_content_type(client: TestClient) -> None:
    response = client.post(
        "/v1/detect/upload",
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415


def test_rejects_empty_upload(client: TestClient) -> None:
    response = client.post(
        "/v1/detect/upload",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400


def test_rejects_oversized_upload(client: TestClient) -> None:
    response = client.post(
        "/v1/detect/upload",
        files={"file": ("large.jpg", b"x" * 1025, "image/jpeg")},
    )

    assert response.status_code == 413
