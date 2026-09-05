"""Configuration and provider registry for swapping adapters at composition time."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import os
from typing import TypeVar


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """The canonical System 1 microphone format and realtime queue limits."""

    device: int | str | None = None
    sample_rate: int = 16_000
    channels: int = 1
    frame_duration_ms: int = 20
    queue_size: int = 50
    turn_buffer_seconds: float = 30.0
    pre_roll_ms: int = 250

    def __post_init__(self) -> None:
        if self.sample_rate != 16_000 or self.channels != 1:
            raise ValueError("System 1 internally accepts only 16 kHz mono audio")
        if not 10 <= self.frame_duration_ms <= 60:
            raise ValueError("frame_duration_ms must stay in the realtime range (10-60 ms)")
        if self.queue_size < 1 or self.turn_buffer_seconds <= 0:
            raise ValueError("audio queue and turn buffer limits must be positive")
        if not self.frame_duration_ms <= self.pre_roll_ms <= 1_000:
            raise ValueError("pre_roll_ms must be at least one frame and no more than one second")

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate * self.frame_duration_ms // 1_000


@dataclass(frozen=True, slots=True)
class VADConfig:
    speech_threshold: float = 0.015
    minimum_speech_ms: int = 60
    minimum_silence_ms: int = 500


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    endpoint_timeout_ms: int = 600
    require_final_transcript: bool = True

    def __post_init__(self) -> None:
        if not 100 <= self.endpoint_timeout_ms <= 5_000:
            raise ValueError("endpoint_timeout_ms must be between 100 and 5000")
        if not self.require_final_transcript:
            raise ValueError("M3A endpointing requires a final transcript for the latest segment")


@dataclass(frozen=True, slots=True)
class NemotronSTTConfig:
    ws_url: str = "ws://localhost:6010/v1/stt/stream"
    health_url: str = "http://localhost:6010/health"
    send_queue_size: int = 100
    connect_timeout_seconds: float = 3.0
    ready_timeout_seconds: float = 5.0
    reconnect_initial_seconds: float = 0.25
    reconnect_max_seconds: float = 3.0

    def __post_init__(self) -> None:
        if self.send_queue_size < 1:
            raise ValueError("STT send_queue_size must be positive")
        if self.connect_timeout_seconds <= 0 or self.ready_timeout_seconds <= 0:
            raise ValueError("STT connection timeouts must be positive")


@dataclass(frozen=True, slots=True)
class System1Config:
    stt_provider: str = "mock.stt"
    dialogue_provider: str = "mock.dialogue"
    tts_provider: str = "mock.tts"
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    nemotron_stt: NemotronSTTConfig = field(default_factory=NemotronSTTConfig)

    @classmethod
    def from_env(cls, **overrides: object) -> "System1Config":
        """Load only deployment-facing provider settings from the environment."""
        provider = os.getenv("GHOSTPILOT_STT_PROVIDER", "mock")
        defaults = NemotronSTTConfig()
        nemotron = NemotronSTTConfig(
            ws_url=os.getenv("GHOSTPILOT_STT_WS_URL", defaults.ws_url),
            health_url=os.getenv("GHOSTPILOT_STT_HEALTH_URL", defaults.health_url),
        )
        values: dict[str, object] = {
            "stt_provider": provider,
            "nemotron_stt": nemotron,
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]


T = TypeVar("T")


class ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], object]] = {}

    def register(self, name: str, factory: Callable[[], T]) -> None:
        self._factories[name] = factory

    def build(self, name: str) -> T:
        try:
            return self._factories[name]()  # type: ignore[return-value]
        except KeyError as error:
            raise ValueError(f"unknown provider: {name}") from error
