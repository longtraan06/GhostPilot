"""Separate, replaceable policy for deciding when a user turn may commit."""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from .config import EndpointConfig
from .transcript import TranscriptSnapshot


class EndpointState(str, Enum):
    IDLE = "IDLE"
    WAITING = "WAITING"
    COMMITTED = "COMMITTED"


class EndpointDetector(Protocol):
    def should_commit(self, transcript: TranscriptSnapshot) -> bool: ...


class FinalTranscriptEndpointDetector:
    """M3A heuristic: only a stable final transcript can end a turn."""

    def __init__(self, config: EndpointConfig) -> None:
        self.config = config

    def should_commit(self, transcript: TranscriptSnapshot) -> bool:
        return bool(transcript.final) if self.config.require_final_transcript else bool(transcript.best)
