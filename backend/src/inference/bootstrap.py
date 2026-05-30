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
        inference_engine=create_inference_engine(resolved),
        media_processor=create_media_processor(resolved),
    )


def create_inference_engine(settings: Settings) -> AbstractInferenceEngine:
    return YoloInferenceEngine(
        model_path=settings.model_path,
        confidence_threshold=settings.confidence_threshold,
    )


def create_media_processor(settings: Settings) -> AbstractMediaProcessor:
    return OpenCvMediaProcessor(
        sample_interval_seconds=settings.video_sample_interval_seconds,
        max_video_frames=settings.max_video_frames,
    )
