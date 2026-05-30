from __future__ import annotations

from dataclasses import dataclass, field

from inference.domain.model import (
    BoundingBox,
    Detection,
    DetectionLabel,
    InferenceResult,
    MediaFrame,
    MediaKind,
    ProcessedMedia,
)
from inference.service_layer.errors import UnsupportedMediaTypeError
from inference.service_layer.media import MediaFormat
from inference.service_layer.ports import AbstractInferenceEngine, AbstractMediaProcessor


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

SAMPLE_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


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


def _empty_media_call_log() -> list[tuple[bytes, MediaFormat]]:
    return []


@dataclass
class FakeMediaProcessor(AbstractMediaProcessor):
    calls: list[tuple[bytes, MediaFormat]] = field(default_factory=_empty_media_call_log)

    def process(self, media_bytes: bytes, media_format: MediaFormat) -> ProcessedMedia:
        self.calls.append((media_bytes, media_format))
        if media_format in {MediaFormat.JPEG, MediaFormat.PNG}:
            return ProcessedMedia(
                kind=MediaKind.IMAGE,
                frames=(
                    MediaFrame(
                        frame_index=0,
                        timestamp_seconds=None,
                        image_bytes=media_bytes,
                    ),
                ),
            )
        if media_format in {
            MediaFormat.MP4,
            MediaFormat.QUICKTIME,
            MediaFormat.AVI,
            MediaFormat.WEBM,
        }:
            return ProcessedMedia(
                kind=MediaKind.VIDEO,
                frames=(
                    MediaFrame(
                        frame_index=0,
                        timestamp_seconds=0.0,
                        image_bytes=b"frame-0",
                    ),
                    MediaFrame(
                        frame_index=8,
                        timestamp_seconds=0.25,
                        image_bytes=b"frame-1",
                    ),
                ),
                sample_interval_seconds=0.25,
            )
        raise UnsupportedMediaTypeError(f"unsupported media format: {media_format}")
