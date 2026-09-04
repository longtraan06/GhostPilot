"""Vendor-neutral provider and local playback contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class STTEvent:
    text: str
    is_final: bool


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
    def events(self) -> AsyncIterator[STTEvent]: ...
    async def close(self) -> None: ...


class DialogueProvider(Protocol):
    def stream(self, transcript: str) -> AsyncIterator[DialogueOutput]: ...
    async def cancel(self) -> None: ...


class TTSProvider(Protocol):
    def stream(self, text: str) -> AsyncIterator[AudioChunk]: ...
    async def cancel(self) -> None: ...


class Playback(Protocol):
    async def play(self, audio: AudioChunk) -> None: ...
    def stop_now(self) -> None: ...
