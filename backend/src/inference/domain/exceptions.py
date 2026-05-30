from __future__ import annotations


class DomainError(Exception):
    """Base exception for framework-free domain failures."""


class InvalidBoundingBox(DomainError):
    """Raised when bounding-box coordinates are structurally invalid."""


class InvalidDetection(DomainError):
    """Raised when a detection value object violates invariants."""
