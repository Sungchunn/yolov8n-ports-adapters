from __future__ import annotations

from dataclasses import dataclass

from inference.adapters.inference import YoloInferenceEngine
from inference.adapters.media import OpenCvMediaProcessor
from inference.config import Settings, get_settings
from inference.service_layer.ports import AbstractInferenceEngine, AbstractMediaProcessor


@dataclass(frozen=True)
class AppDependencies:
    inference_engine: AbstractInferenceEngine
    media_processor: AbstractMediaProcessor


def bootstrap(settings: Settings | None = None) -> AppDependencies:
    resolved = settings or get_settings()
    return AppDependencies(
        inference_engine=YoloInferenceEngine(
            model_path=resolved.model_path,
            confidence_threshold=resolved.confidence_threshold,
        ),
        media_processor=OpenCvMediaProcessor(
            sample_interval_seconds=resolved.video_sample_interval_seconds,
            max_video_frames=resolved.max_video_frames,
        ),
    )
