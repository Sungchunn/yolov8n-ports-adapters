from __future__ import annotations

from typing import Any

import pytest

from inference.adapters.media import OpenCvMediaProcessor, SampledFrame, sample_video_capture
from inference.domain.exceptions import InvalidImageError, UnsupportedMediaTypeError
from inference.domain.model import MediaKind
from tests.conftest import SAMPLE_JPEG_BYTES, SAMPLE_PNG_BYTES


class FakeCapture:
    def __init__(self, frames: list[str]) -> None:
        self.frames = frames
        self.read_count = 0

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, Any]:
        if self.read_count >= len(self.frames):
            return False, None
        frame = self.frames[self.read_count]
        self.read_count += 1
        return True, frame

    def get(self, _prop_id: int) -> float:
        return 0

    def release(self) -> None:
        return None


def test_sample_video_capture_uses_interval_and_max_frames() -> None:
    capture = FakeCapture(["a", "b", "c", "d", "e", "f"])

    frames = list(
        sample_video_capture(
            capture,
            sample_interval_seconds=0.5,
            max_frames=2,
            encode_frame=lambda frame: f"jpeg-{frame}".encode(),
            fps=4,
        )
    )

    assert [frame.frame_index for frame in frames] == [0, 2]
    assert [frame.timestamp_seconds for frame in frames] == [0, 0.5]
    assert [frame.image_bytes for frame in frames] == [b"jpeg-a", b"jpeg-c"]


def test_media_processor_accepts_jpeg_image() -> None:
    processor = OpenCvMediaProcessor(sample_interval_seconds=1.0, max_video_frames=60)

    media = processor.process(SAMPLE_JPEG_BYTES, "image/jpeg")

    assert media.kind == MediaKind.IMAGE
    assert media.frames[0].image_bytes == SAMPLE_JPEG_BYTES


def test_media_processor_accepts_png_image() -> None:
    processor = OpenCvMediaProcessor(sample_interval_seconds=1.0, max_video_frames=60)

    media = processor.process(SAMPLE_PNG_BYTES, "image/png")

    assert media.kind == MediaKind.IMAGE
    assert media.frames[0].image_bytes == SAMPLE_PNG_BYTES


def test_media_processor_rejects_invalid_image_magic() -> None:
    processor = OpenCvMediaProcessor(sample_interval_seconds=1.0, max_video_frames=60)

    with pytest.raises(InvalidImageError):
        processor.process(b"not a png", "image/png")


def test_media_processor_dispatches_avi_to_video_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_sample_video_bytes(
        video_bytes: bytes,
        sample_interval_seconds: float,
        max_frames: int,
        suffix: str = ".mp4",
    ) -> list[SampledFrame]:
        assert video_bytes == b"avi bytes"
        assert sample_interval_seconds == 0.5
        assert max_frames == 2
        assert suffix == ".avi"
        return [
            SampledFrame(frame_index=0, timestamp_seconds=0.0, image_bytes=b"frame")
        ]

    monkeypatch.setattr(
        "inference.adapters.media.sample_video_bytes",
        fake_sample_video_bytes,
    )
    processor = OpenCvMediaProcessor(sample_interval_seconds=0.5, max_video_frames=2)

    media = processor.process(b"avi bytes", "video/x-msvideo")

    assert media.kind == MediaKind.VIDEO
    assert media.sample_interval_seconds == 0.5
    assert media.frames[0].image_bytes == b"frame"


def test_media_processor_rejects_unsupported_media_type() -> None:
    processor = OpenCvMediaProcessor(sample_interval_seconds=1.0, max_video_frames=60)

    with pytest.raises(UnsupportedMediaTypeError):
        processor.process(b"hello", "text/plain")
