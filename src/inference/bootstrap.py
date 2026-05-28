from __future__ import annotations

from inference.adapters.inference import YoloInferenceEngine
from inference.config import Settings, get_settings
from inference.service_layer.ports import AbstractInferenceEngine


def bootstrap(settings: Settings | None = None) -> AbstractInferenceEngine:
    resolved = settings or get_settings()
    return YoloInferenceEngine(
        model_path=resolved.model_path,
        confidence_threshold=resolved.confidence_threshold,
    )
