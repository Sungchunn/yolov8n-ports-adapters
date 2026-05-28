from __future__ import annotations


class DomainError(Exception):
    """Base exception for framework-free domain/application failures."""


class InvalidImageError(DomainError):
    """Raised when image bytes cannot be decoded for inference."""


class InvalidVideoError(DomainError):
    """Raised when video bytes cannot be decoded or sampled."""


class InvalidBoundingBox(DomainError):
    """Raised when bounding-box coordinates are structurally invalid."""


class InvalidDetection(DomainError):
    """Raised when a detection value object violates invariants."""


class InferenceError(DomainError):
    """Raised when the concrete inference runtime fails."""
