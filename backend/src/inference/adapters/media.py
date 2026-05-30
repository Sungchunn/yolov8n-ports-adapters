from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
import tempfile
from typing import Any, Callable, Iterator, Protocol, cast

from inference.domain.exceptions import (
    InferenceError,
    InvalidImageError,
    InvalidVideoError,
    UnsupportedMediaTypeError,
)
from inference.domain.model import MediaFrame, MediaKind, ProcessedMedia
from inference.service_layer.ports import AbstractMediaProcessor

IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}
VIDEO_CONTENT_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/webm": ".webm",
}


class CaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Any]: ...

    def get(self, prop_id: int) -> float: ...

    def release(self) -> None: ...


class EncodedImageLike(Protocol):
    def tobytes(self) -> bytes: ...


class Cv2Like(Protocol):
    CAP_PROP_FPS: int

    def VideoCapture(self, source: int | str) -> CaptureLike: ...

    def imencode(self, ext: str, frame: Any) -> tuple[bool, EncodedImageLike]: ...


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    timestamp_seconds: float
    image_bytes: bytes


def _load_cv2() -> Cv2Like:
    try:
        cv2 = import_module("cv2")
    except Exception as exc:  # pragma: no cover - depends on optional runtime extra
        raise InferenceError(
            "opencv-python is required for video upload support; "
            "install with `uv sync --extra vision`"
        ) from exc
    return cast(Cv2Like, cv2)


class OpenCvMediaProcessor(AbstractMediaProcessor):
    def __init__(self, sample_interval_seconds: float, max_video_frames: int) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self.max_video_frames = max_video_frames

    def process(self, media_bytes: bytes, content_type: str) -> ProcessedMedia:
        if not media_bytes:
            raise InvalidImageError("media payload must not be empty")
        if content_type in IMAGE_CONTENT_TYPES:
            self._validate_image(media_bytes, content_type)
            return ProcessedMedia(
                kind=MediaKind.IMAGE,
                frames=(MediaFrame(frame_index=0, timestamp_seconds=None, image_bytes=media_bytes),),
            )
        if content_type in VIDEO_CONTENT_TYPES:
            frames = sample_video_bytes(
                media_bytes,
                sample_interval_seconds=self.sample_interval_seconds,
                max_frames=self.max_video_frames,
                suffix=VIDEO_CONTENT_TYPES[content_type],
            )
            return ProcessedMedia(
                kind=MediaKind.VIDEO,
                frames=tuple(
                    MediaFrame(
                        frame_index=frame.frame_index,
                        timestamp_seconds=frame.timestamp_seconds,
                        image_bytes=frame.image_bytes,
                    )
                    for frame in frames
                ),
                sample_interval_seconds=self.sample_interval_seconds,
            )
        raise UnsupportedMediaTypeError(f"unsupported media type: {content_type}")

    def _validate_image(self, media_bytes: bytes, content_type: str) -> None:
        if content_type in {"image/jpeg", "image/jpg"}:
            if not media_bytes.startswith(b"\xff\xd8\xff"):
                raise InvalidImageError("upload payload is not a JPEG image")
            return
        if content_type == "image/png":
            if not media_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                raise InvalidImageError("upload payload is not a PNG image")
            return
        raise UnsupportedMediaTypeError(f"unsupported image media type: {content_type}")


def open_capture(source: int | str) -> CaptureLike:
    cv2 = _load_cv2()
    return cv2.VideoCapture(source)


def encode_frame_to_jpeg(frame: Any) -> bytes:
    cv2 = _load_cv2()
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise InvalidVideoError("failed to encode frame as JPEG")
    return encoded.tobytes()


def sample_video_capture(
    capture: CaptureLike,
    sample_interval_seconds: float,
    max_frames: int,
    *,
    encode_frame: Callable[[Any], bytes] = encode_frame_to_jpeg,
    fps: float | None = None,
) -> Iterator[SampledFrame]:
    if sample_interval_seconds <= 0:
        raise InvalidVideoError("sample interval must be positive")
    if max_frames <= 0:
        raise InvalidVideoError("max video frames must be positive")
    if not capture.isOpened():
        raise InvalidVideoError("video file could not be opened")

    if fps is None:
        cv2 = _load_cv2()
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 1.0

    frame_interval = max(1, round(fps * sample_interval_seconds))
    read_index = 0
    emitted = 0
    while emitted < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if read_index % frame_interval == 0:
            yield SampledFrame(
                frame_index=read_index,
                timestamp_seconds=read_index / fps,
                image_bytes=encode_frame(frame),
            )
            emitted += 1
        read_index += 1


def sample_video_bytes(
    video_bytes: bytes,
    sample_interval_seconds: float,
    max_frames: int,
    suffix: str = ".mp4",
) -> list[SampledFrame]:
    if not video_bytes:
        raise InvalidVideoError("video payload must not be empty")

    path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_bytes)
            path = temp_file.name

        capture = open_capture(path)
        try:
            frames = list(
                sample_video_capture(
                    capture,
                    sample_interval_seconds=sample_interval_seconds,
                    max_frames=max_frames,
                )
            )
        finally:
            capture.release()
    finally:
        if path:
            os.unlink(path)

    if not frames:
        raise InvalidVideoError("video contained no decodable frames")
    return frames
