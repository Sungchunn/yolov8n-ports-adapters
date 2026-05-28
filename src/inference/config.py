from __future__ import annotations

from dataclasses import dataclass
import os


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "models/yolov8n.pt")
    confidence_threshold: float = _float_from_env("CONFIDENCE_THRESHOLD", 0.25)
    max_upload_bytes: int = _int_from_env("MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
    video_sample_interval_seconds: float = _float_from_env(
        "VIDEO_SAMPLE_INTERVAL_SECONDS", 1.0
    )
    max_video_frames: int = _int_from_env("MAX_VIDEO_FRAMES", 60)
    webcam_index: int = _int_from_env("WEBCAM_INDEX", 0)
    webcam_fps_limit: float = _float_from_env("WEBCAM_FPS_LIMIT", 2.0)


def get_settings() -> Settings:
    return Settings()
