"""Deterministic providers for lifecycle tests and local development."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable

from .providers import AudioChunk, DialogueOutput, STTEvent


class MockSTTProvider:
    def __init__(self) -> None:
        self._events: asyncio.Queue[STTEvent | None] = asyncio.Queue()
        self.connected = False
        self.audio_frames: list[bytes] = []

    async def connect(self) -> None:
        self.connected = True

    async def send_audio(self, audio: bytes) -> None:
        if not self.connected:
            raise RuntimeError("STT provider is not connected")
        self.audio_frames.append(audio)

    async def emit(self, text: str, *, is_final: bool, turn_id: str | None = None) -> None:
        await self._events.put(STTEvent(text, is_final, turn_id))

    async def events(self) -> AsyncIterator[STTEvent]:
        while (event := await self._events.get()) is not None:
            yield event

    async def close(self) -> None:
        self.connected = False
        await self._events.put(None)


class MockDialogueProvider:
    def __init__(self, responses: Iterable[DialogueOutput] | None = None, *, delay: float = 0) -> None:
        self.responses = list(responses or [DialogueOutput("Okay. I am ready to help.")])
        self.delay = delay
        self.cancelled = False
        self.stream_calls = 0

    async def stream(self, transcript: str) -> AsyncIterator[DialogueOutput]:
        self.stream_calls += 1
        self.cancelled = False
        for response in self.responses:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.cancelled:
                return
            yield response

    async def cancel(self) -> None:
        self.cancelled = True


class MockTTSProvider:
    def __init__(self, *, delay: float = 0) -> None:
        self.delay = delay
        self.cancelled = False
        self.stream_calls = 0

    async def stream(self, text: str) -> AsyncIterator[AudioChunk]:
        self.stream_calls += 1
        self.cancelled = False
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.cancelled:
            yield AudioChunk(text.encode())

    async def cancel(self) -> None:
        self.cancelled = True


class MockPlayback:
    def __init__(self) -> None:
        self.played: list[AudioChunk] = []
        self.stopped = False

    async def play(self, audio: AudioChunk) -> None:
        # A new stream owns a fresh playback buffer after a previous stop.
        self.stopped = False
        self.played.append(audio)

    def stop_now(self) -> None:
        self.stopped = True
