"""Deterministic audio and VAD fakes for System 1 tests."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterable

from .audio import AudioFrame
from .vad import VADEvent, VADState


class FakeAudioInput:
    def __init__(self, *, queue_size: int = 16) -> None:
        self._queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue(queue_size)
        self.frames_dropped = 0
        self.started = False

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        self.started = True

    def push(self, frame: AudioFrame) -> None:
        if self._queue.full():
            self.frames_dropped += 1
            return
        self._queue.put_nowait(frame)

    async def frames(self) -> AsyncIterator[AudioFrame]:
        while (frame := await self._queue.get()) is not None:
            yield frame

    async def close(self) -> None:
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(None)


class FakeVAD:
    def __init__(self, events: Iterable[VADEvent | None]) -> None:
        self._events = deque(events)
        self._state = VADState.LISTENING
        self._last_probability: float | None = None

    @property
    def state(self) -> VADState:
        return self._state

    @property
    def last_probability(self) -> float | None:
        return self._last_probability

    def process(self, frame: AudioFrame) -> VADEvent | None:
        event = self._events.popleft() if self._events else None
        if event:
            self._last_probability = event.probability
            self._state = (
                VADState.SPEAKING if event.kind.value == "SPEECH_STARTED" else VADState.LISTENING
            )
        return event
