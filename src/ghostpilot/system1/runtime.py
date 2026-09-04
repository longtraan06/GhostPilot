"""Minimal System 1 composition root."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import cast

from .audio import AudioInput, TurnAudioBuffer
from .config import ProviderRegistry, System1Config
from .event_bus import EventBus
from .events import AudioFrameDropped, AudioInputStarted, AudioInputStopped
from .interruption import InterruptionController
from .mock_providers import MockDialogueProvider, MockPlayback, MockSTTProvider, MockTTSProvider
from .providers import DialogueProvider, Playback, STTProvider, TTSProvider
from .state import ConversationState
from .turn import TurnManager
from .vad import VADEventKind, VoiceActivityDetector, VADState, pcm16_rms_level


class System1Runtime:
    def __init__(
        self,
        *,
        stt: STTProvider | None = None,
        dialogue: DialogueProvider | None = None,
        tts: TTSProvider | None = None,
        playback: Playback | None = None,
        audio_input: AudioInput | None = None,
        vad: VoiceActivityDetector | None = None,
        config: System1Config | None = None,
    ) -> None:
        if (audio_input is None) != (vad is None):
            raise ValueError("audio_input and vad must be provided together")
        self.config = config or System1Config()
        self.events = EventBus()
        self.state = ConversationState()
        self.stt = stt or MockSTTProvider()
        self.dialogue = dialogue or MockDialogueProvider()
        self.tts = tts or MockTTSProvider()
        self.playback = playback or MockPlayback()
        self.audio_input, self.vad = audio_input, vad
        self.turn_audio_buffer: TurnAudioBuffer | None = None
        self._audio_task: asyncio.Task[None] | None = None
        self._started = False
        self._audio_level = 0.0
        self._reported_dropped_frames = 0
        self.interruption = InterruptionController(
            self.state, self.events, self.dialogue, self.tts, self.playback
        )
        self.turns = TurnManager(
            self.state, self.events, self.dialogue, self.tts, self.playback, self.interruption
        )

    @classmethod
    def from_config(
        cls,
        config: System1Config,
        registry: ProviderRegistry,
        *,
        playback: Playback | None = None,
    ) -> "System1Runtime":
        """Compose adapters selected by names, not by domain-code imports."""
        return cls(
            stt=cast(STTProvider, registry.build(config.stt_provider)),
            dialogue=cast(DialogueProvider, registry.build(config.dialogue_provider)),
            tts=cast(TTSProvider, registry.build(config.tts_provider)),
            playback=playback,
            config=config,
        )

    async def start(self) -> None:
        await self.stt.connect()
        self._started = True
        if self.audio_input:
            await self._start_audio_input()

    async def close(self) -> None:
        await self._stop_audio_input()
        await self.stt.close()
        self._started = False

    async def configure_audio_input(
        self,
        audio_input: AudioInput,
        vad: VoiceActivityDetector,
        *,
        config: System1Config,
    ) -> None:
        """Swap the local input adapter without coupling runtime to a device vendor."""
        if not self._started:
            raise RuntimeError("start the System 1 runtime before configuring audio")
        await self._stop_audio_input()
        self.audio_input, self.vad, self.config = audio_input, vad, config
        self._reported_dropped_frames = 0
        try:
            await self._start_audio_input()
        except Exception:
            # Do not leave a failed adapter on the active realtime path.
            self.audio_input, self.vad = None, None
            raise

    async def _start_audio_input(self) -> None:
        assert self.audio_input is not None
        await self.audio_input.start()
        await self.events.publish(AudioInputStarted())
        self._audio_task = asyncio.create_task(self._run_audio_loop())

    async def _stop_audio_input(self) -> None:
        if self._audio_task:
            self._audio_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._audio_task
            self._audio_task = None
        if self.audio_input:
            await self.audio_input.close()
            await self.events.publish(AudioInputStopped())

    async def on_user_speech_started(self) -> str:
        turn_id = await self.turns.user_speech_started()
        if self.turn_audio_buffer is None or self.turn_audio_buffer.turn_id != turn_id:
            self.turn_audio_buffer = TurnAudioBuffer(
                turn_id, maximum_seconds=self.config.audio.turn_buffer_seconds
            )
        return turn_id

    async def on_user_speech_stopped(self) -> None:
        """Record a VAD boundary; endpoint detection decides when to commit."""
        await self.turns.user_speech_stopped()

    async def commit_turn(self, transcript: str) -> None:
        """Start response generation after endpoint detection accepts this turn."""
        await self.turns.commit_turn(transcript)

    async def wait_for_response(self) -> None:
        await self.turns.wait_for_response()

    async def _run_audio_loop(self) -> None:
        assert self.audio_input is not None and self.vad is not None
        async for frame in self.audio_input.frames():
            was_user_speaking = self.state.user_state.value == "SPEAKING"
            if was_user_speaking and self.turn_audio_buffer:
                self.turn_audio_buffer.append(frame)
            self._audio_level = pcm16_rms_level(frame)
            vad_event = self.vad.process(frame)
            if vad_event and vad_event.kind is VADEventKind.SPEECH_STARTED:
                await self.on_user_speech_started()
                if not was_user_speaking and self.turn_audio_buffer:
                    self.turn_audio_buffer.append(frame)
            elif vad_event and vad_event.kind is VADEventKind.SPEECH_STOPPED:
                await self.on_user_speech_stopped()
            await self._report_input_drops()

    async def _report_input_drops(self) -> None:
        if self.audio_input is None:
            return
        dropped = getattr(self.audio_input, "frames_dropped", 0)
        if dropped > self._reported_dropped_frames:
            self._reported_dropped_frames = dropped
            await self.events.publish(AudioFrameDropped(dropped))

    def debug_snapshot(self) -> dict[str, object]:
        """Small state only: debug tooling never receives raw microphone PCM."""
        return {
            "vad_state": self.vad.state.value if self.vad else VADState.LISTENING.value,
            "turn_state": self.state.turn_state.value,
            "audio_level": round(self._audio_level, 3),
            "vad_probability": self.vad.last_probability if self.vad else None,
            "frames_dropped": getattr(self.audio_input, "frames_dropped", 0),
            "audio_queue_size": getattr(self.audio_input, "queue_size", 0),
            "audio_device": self.config.audio.device,
            "audio_connected": self._audio_task is not None and not self._audio_task.done(),
            "sample_rate": self.config.audio.sample_rate,
            "frame_duration_ms": self.config.audio.frame_duration_ms,
            "buffered_audio_seconds": round(
                self.turn_audio_buffer.duration_seconds if self.turn_audio_buffer else 0.0, 3
            ),
        }


def default_provider_registry() -> ProviderRegistry:
    """Development-only composition; production registers adapter factories here."""
    registry = ProviderRegistry()
    registry.register("mock.stt", MockSTTProvider)
    registry.register("mock.dialogue", MockDialogueProvider)
    registry.register("mock.tts", MockTTSProvider)
    return registry
