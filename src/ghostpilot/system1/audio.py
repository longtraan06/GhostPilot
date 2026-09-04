"""Canonical realtime audio transport types and bounded turn buffering."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


CANONICAL_SAMPLE_RATE = 16_000
CANONICAL_CHANNELS = 1
CANONICAL_SAMPLE_WIDTH_BYTES = 2  # PCM16 / signed int16


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """A canonical mono PCM16 frame on System 1's realtime data path."""

    data: bytes
    sample_rate: int
    channels: int
    timestamp: float
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.sample_rate != CANONICAL_SAMPLE_RATE or self.channels != CANONICAL_CHANNELS:
            raise ValueError("AudioFrame must use canonical 16 kHz mono audio")
        if len(self.data) % CANONICAL_SAMPLE_WIDTH_BYTES:
            raise ValueError("PCM16 frame data must contain complete int16 samples")

    @property
    def samples(self) -> int:
        return len(self.data) // CANONICAL_SAMPLE_WIDTH_BYTES // self.channels

    @property
    def duration_seconds(self) -> float:
        return self.samples / self.sample_rate


class AudioInput(Protocol):
    async def start(self) -> None: ...
    def frames(self) -> AsyncIterator[AudioFrame]: ...
    async def close(self) -> None: ...


class TurnAudioBuffer:
    """Bounded audio retained for the currently uncommitted user turn."""

    def __init__(self, turn_id: str, *, maximum_seconds: float) -> None:
        self.turn_id = turn_id
        self._maximum_seconds = maximum_seconds
        self._frames: deque[AudioFrame] = deque()
        self._duration_seconds = 0.0
        self.dropped_frames = 0

    def append(self, frame: AudioFrame) -> None:
        self._frames.append(frame)
        self._duration_seconds += frame.duration_seconds
        while self._frames and self._duration_seconds > self._maximum_seconds:
            removed = self._frames.popleft()
            self._duration_seconds -= removed.duration_seconds
            self.dropped_frames += 1

    @property
    def duration_seconds(self) -> float:
        return self._duration_seconds

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def snapshot(self) -> tuple[AudioFrame, ...]:
        """A stable view for the future STT adapter; frames themselves are not copied."""
        return tuple(self._frames)
