"""Vendor-neutral provider and local playback contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True, slots=True)
class STTEvent:
    text: str
    is_final: bool
    turn_id: str | None = None
    # This is a provider-neutral association, not a vendor result ID.  An
    # adapter may attach the speech segment that was active when it submitted
    # audio so a delayed result cannot update a later segment of the same turn.
    segment_id: int | None = None


@dataclass(frozen=True, slots=True)
class STTServiceEvent:
    """Provider-neutral lifecycle/error signal from a streaming STT adapter."""

    status: Literal["connecting", "connected", "ready", "disconnected", "error", "reset"]
    detail: str = ""
    connection_generation: int = 0


STTProviderEvent = STTEvent | STTServiceEvent


@dataclass(frozen=True, slots=True)
class DialogueOutput:
    text: str = ""
    action: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AudioChunk:
    data: bytes


class STTProvider(Protocol):
    async def connect(self) -> None: ...
    async def send_audio(self, audio: bytes) -> None: ...
    async def start_segment(self, turn_id: str, segment_id: int) -> None: ...
    async def end_segment(self) -> None: ...
    async def commit_turn(self) -> None: ...
    async def reset(self) -> None: ...
    async def ping(self) -> None: ...
    async def reconnect(self) -> None: ...
    def events(self) -> AsyncIterator[STTProviderEvent]: ...
    def diagnostics(self) -> dict[str, object]: ...
    async def close(self) -> None: ...


class DialogueProvider(Protocol):
    def stream(self, transcript: str) -> AsyncIterator[DialogueOutput]: ...
    async def cancel(self) -> None: ...


class TTSProvider(Protocol):
    def stream(self, text: str) -> AsyncIterator[AudioChunk]: ...
    async def cancel(self) -> None: ...


class Playback(Protocol):
    async def play(self, audio: AudioChunk) -> None: ...
    def stop_now(self) -> None:
        """Stop only current playback; a later ``play`` call must still be accepted."""
