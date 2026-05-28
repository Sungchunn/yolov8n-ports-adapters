from __future__ import annotations

from inference.domain.exceptions import InvalidImageError
from inference.domain.model import InferenceResult
from inference.service_layer.ports import AbstractInferenceEngine


def detect_objects(
    image_bytes: bytes, engine: AbstractInferenceEngine
) -> InferenceResult:
    if not image_bytes:
        raise InvalidImageError("image payload must not be empty")
    return engine.predict(image_bytes)
