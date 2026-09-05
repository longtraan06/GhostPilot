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
        # A final from a previous VAD speech segment belongs to the same user
        # turn, but must never authorize committing a later resumed segment.
        return transcript.latest_segment_final and bool(transcript.commit_text)
