from __future__ import annotations

from abc import ABC, abstractmethod

from inference.domain.model import InferenceResult, ProcessedMedia


class AbstractInferenceEngine(ABC):
    @abstractmethod
    def predict(self, image_bytes: bytes) -> InferenceResult:
        """Run one-frame inference and return a domain result."""


class AbstractMediaProcessor(ABC):
    @abstractmethod
    def process(self, media_bytes: bytes, content_type: str) -> ProcessedMedia:
        """Convert uploaded media into inference-ready image frames."""
