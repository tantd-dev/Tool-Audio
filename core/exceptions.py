"""
Custom exceptions for ToolAudio.
"""

from __future__ import annotations


class MethodError(Exception):
    """Exception raised during benchmark method execution with additional diagnostic data."""
    def __init__(self, msg: str, extra: dict | None = None):
        super().__init__(msg)
        self.extra = extra or {}
