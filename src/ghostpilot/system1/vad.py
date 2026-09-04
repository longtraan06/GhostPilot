"""Vendor-neutral local voice activity detection."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Protocol

from .audio import AudioFrame
from .config import VADConfig


class VADState(str, Enum):
    LISTENING = "LISTENING"
    SPEAKING = "SPEAKING"


class VADEventKind(str, Enum):
    SPEECH_STARTED = "SPEECH_STARTED"
    SPEECH_STOPPED = "SPEECH_STOPPED"


@dataclass(frozen=True, slots=True)
class VADEvent:
    kind: VADEventKind
    probability: float | None = None


class VoiceActivityDetector(Protocol):
    @property
    def state(self) -> VADState: ...
    @property
    def last_probability(self) -> float | None: ...
    def process(self, frame: AudioFrame) -> VADEvent | None: ...


def pcm16_rms_level(frame: AudioFrame) -> float:
    """Return a debug-only normalized RMS level in approximately the 0..1 range."""
    samples = array("h")
    samples.frombytes(frame.data)
    if not samples:
        return 0.0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    return min(1.0, sqrt(mean_square) / 32_768.0)


class EnergyVoiceActivityDetector:
    """Lightweight local baseline; replaceable with a Silero adapter later."""

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self._state = VADState.LISTENING
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0
        self._last_probability: float | None = None

    @property
    def state(self) -> VADState:
        return self._state

    @property
    def last_probability(self) -> float | None:
        return self._last_probability

    def process(self, frame: AudioFrame) -> VADEvent | None:
        level = pcm16_rms_level(frame)
        probability = min(1.0, level / self.config.speech_threshold)
        self._last_probability = probability
        duration = frame.duration_seconds
        speech = level >= self.config.speech_threshold

        if self._state is VADState.LISTENING:
            self._speech_seconds = self._speech_seconds + duration if speech else 0.0
            if self._speech_seconds >= self.config.minimum_speech_ms / 1_000:
                self._state = VADState.SPEAKING
                self._silence_seconds = 0.0
                return VADEvent(VADEventKind.SPEECH_STARTED, probability)
            return None

        self._silence_seconds = self._silence_seconds + duration if not speech else 0.0
        if self._silence_seconds >= self.config.minimum_silence_ms / 1_000:
            self._state = VADState.LISTENING
            self._speech_seconds = 0.0
            return VADEvent(VADEventKind.SPEECH_STOPPED, probability)
        return None
