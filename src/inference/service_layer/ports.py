from __future__ import annotations

from abc import ABC, abstractmethod

from inference.domain.model import InferenceResult


class AbstractInferenceEngine(ABC):
    @abstractmethod
    def predict(self, image_bytes: bytes) -> InferenceResult:
        """Run one-frame inference and return a domain result."""
