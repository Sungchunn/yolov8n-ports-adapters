from __future__ import annotations

from dataclasses import dataclass
import os


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _list_from_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "models/yolov8n.pt")
    confidence_threshold: float = _float_from_env("CONFIDENCE_THRESHOLD", 0.25)
    max_upload_bytes: int = _int_from_env("MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
    video_sample_interval_seconds: float = _float_from_env(
        "VIDEO_SAMPLE_INTERVAL_SECONDS", 0.25
    )
    max_video_frames: int = _int_from_env("MAX_VIDEO_FRAMES", 120)
    cors_origins: tuple[str, ...] = _list_from_env(
        "CORS_ORIGINS",
        ("http://localhost:3000", "http://127.0.0.1:3000"),
    )


def get_settings() -> Settings:
    return Settings()
