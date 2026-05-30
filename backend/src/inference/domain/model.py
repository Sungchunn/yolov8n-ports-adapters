from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from inference.domain.exceptions import InvalidBoundingBox, InvalidDetection


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if any(value < 0 for value in values):
            raise InvalidBoundingBox("bounding box coordinates must be non-negative")
        if self.x2 < self.x1:
            raise InvalidBoundingBox("x2 must be greater than or equal to x1")
        if self.y2 < self.y1:
            raise InvalidBoundingBox("y2 must be greater than or equal to y1")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.width / 2, self.y1 + self.height / 2)


@dataclass(frozen=True)
class DetectionLabel:
    class_id: int
    name: str

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise InvalidDetection("class_id must be non-negative")
        if not self.name.strip():
            raise InvalidDetection("label name must not be blank")


@dataclass(frozen=True)
class Detection:
    box: BoundingBox
    label: DetectionLabel
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise InvalidDetection("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class InferenceResult:
    detections: Sequence[Detection]
    image_width: int
    image_height: int
    inference_ms: float
    model_name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise InvalidDetection("image dimensions must be positive")
        if self.inference_ms < 0:
            raise InvalidDetection("inference_ms must be non-negative")
        if not self.model_name.strip():
            raise InvalidDetection("model_name must not be blank")
        if not isinstance(self.detections, tuple):
            object.__setattr__(self, "detections", tuple(self.detections))

    def __len__(self) -> int:
        return len(self.detections)

    def count(self) -> int:
        return len(self)


class MediaKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True)
class MediaFrame:
    frame_index: int
    timestamp_seconds: float | None
    image_bytes: bytes

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise InvalidDetection("frame_index must be non-negative")
        if self.timestamp_seconds is not None and self.timestamp_seconds < 0:
            raise InvalidDetection("timestamp_seconds must be non-negative")
        if not self.image_bytes:
            raise InvalidDetection("frame image bytes must not be empty")


@dataclass(frozen=True)
class ProcessedMedia:
    kind: MediaKind
    frames: Sequence[MediaFrame]
    sample_interval_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.frames:
            raise InvalidDetection("processed media must contain at least one frame")
        if not isinstance(self.frames, tuple):
            object.__setattr__(self, "frames", tuple(self.frames))
        if self.kind == MediaKind.IMAGE and len(self.frames) != 1:
            raise InvalidDetection("image media must contain exactly one frame")
        if self.kind == MediaKind.VIDEO:
            if self.sample_interval_seconds is None:
                raise InvalidDetection("video media must include a sample interval")
            if self.sample_interval_seconds <= 0:
                raise InvalidDetection("sample interval must be positive")
