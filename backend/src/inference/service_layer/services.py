from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inference.domain.exceptions import InvalidDetection
from inference.domain.model import InferenceResult, MediaKind
from inference.service_layer.errors import InvalidImageError
from inference.service_layer.media import MediaFormat
from inference.service_layer.ports import AbstractInferenceEngine, AbstractMediaProcessor


@dataclass(frozen=True)
class DetectionFrameResult:
    frame_index: int
    timestamp_seconds: float | None
    result: InferenceResult


@dataclass(frozen=True)
class MediaDetectionResult:
    kind: MediaKind
    frames: Sequence[DetectionFrameResult]
    sample_interval_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.frames:
            raise InvalidDetection("media detection result must contain at least one frame")
        if not isinstance(self.frames, tuple):
            object.__setattr__(self, "frames", tuple(self.frames))
        if self.kind == MediaKind.IMAGE and len(self.frames) != 1:
            raise InvalidDetection("image detection result must contain exactly one frame")
        if self.kind == MediaKind.VIDEO and self.sample_interval_seconds is None:
            raise InvalidDetection("video detection result must include a sample interval")


def detect_objects(
    image_bytes: bytes, engine: AbstractInferenceEngine
) -> InferenceResult:
    if not image_bytes:
        raise InvalidImageError("image payload must not be empty")
    return engine.predict(image_bytes)


def detect_media(
    media_bytes: bytes,
    media_format: MediaFormat,
    inference_engine: AbstractInferenceEngine,
    media_processor: AbstractMediaProcessor,
) -> MediaDetectionResult:
    if not media_bytes:
        raise InvalidImageError("media payload must not be empty")

    processed_media = media_processor.process(media_bytes, media_format)
    return MediaDetectionResult(
        kind=processed_media.kind,
        frames=tuple(
            DetectionFrameResult(
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                result=detect_objects(frame.image_bytes, inference_engine),
            )
            for frame in processed_media.frames
        ),
        sample_interval_seconds=processed_media.sample_interval_seconds,
    )
