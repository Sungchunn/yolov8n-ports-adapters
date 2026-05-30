from __future__ import annotations


class ApplicationError(Exception):
    """Base exception for framework-free application failures."""


class InvalidImageError(ApplicationError):
    """Raised when image bytes cannot be decoded for inference."""


class InvalidVideoError(ApplicationError):
    """Raised when video bytes cannot be decoded or sampled."""


class UnsupportedMediaTypeError(ApplicationError):
    """Raised when uploaded media has no supported processing adapter."""


class InferenceError(ApplicationError):
    """Raised when the concrete inference runtime fails."""
