"""Minimal System 1 composition root."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from contextlib import suppress
import logging
from typing import cast

from .audio import AudioFrame, AudioInput, AudioPreRollBuffer, TurnAudioBuffer
from .config import ProviderRegistry, System1Config
from .endpoint import EndpointDetector, EndpointState, FinalTranscriptEndpointDetector
from .event_bus import EventBus
from .events import (
    AudioFrameDropped,
    AudioInputStarted,
    AudioInputStopped,
    ProviderFailed,
    ProviderStatusChanged,
    TranscriptFinal,
    TranscriptPartial,
)
from .interruption import InterruptionController
from .mock_providers import MockDialogueProvider, MockPlayback, MockSTTProvider, MockTTSProvider
from .providers import (
    DialogueProvider,
    Playback,
    STTEvent,
    STTProvider,
    STTServiceEvent,
    TTSProvider,
)
from .state import ConversationState, TurnState
from .turn import TurnManager
from .transcript import TranscriptManager
from .vad import VADEventKind, VoiceActivityDetector, VADState, pcm16_rms_level


logger = logging.getLogger(__name__)


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
        endpoint_detector: EndpointDetector | None = None,
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
        self.pre_roll_buffer = AudioPreRollBuffer(
            maximum_seconds=self.config.audio.pre_roll_ms / 1_000
        )
        self._audio_task: asyncio.Task[None] | None = None
        self._stt_task: asyncio.Task[None] | None = None
        self._endpoint_task: asyncio.Task[None] | None = None
        self._endpoint_generation = 0
        self._endpoint_deadline: float | None = None
        self._started = False
        self._audio_level = 0.0
        self._reported_dropped_frames = 0
        self._speech_started_at: float | None = None
        self._speech_stopped_at: float | None = None
        self._first_partial_latency_ms: float | None = None
        self._segment_final_latency_ms: float | None = None
        self._turn_commit_latency_ms: float | None = None
        self._last_stt_response_at: float | None = None
        self._last_committed_turn_id: str | None = None
        self._latency_history: deque[dict[str, object]] = deque(maxlen=10)
        self._audio_frame_observer: Callable[[AudioFrame], None] | None = None
        self.transcripts = TranscriptManager()
        self.endpoint_detector = endpoint_detector or FinalTranscriptEndpointDetector(
            self.config.endpoint
        )
        self.endpoint_state = EndpointState.IDLE
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
        self._stt_task = asyncio.create_task(self._run_stt_loop())
        if self.audio_input:
            await self._start_audio_input()

    async def close(self) -> None:
        await self._cancel_pending_endpoint()
        await self._stop_audio_input()
        if self._stt_task:
            self._stt_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._stt_task
            self._stt_task = None
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
        self.pre_roll_buffer = AudioPreRollBuffer(
            maximum_seconds=self.config.audio.pre_roll_ms / 1_000
        )
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
        resuming = self.state.turn_state is TurnState.AWAITING_COMMIT
        await self._cancel_pending_endpoint()
        turn_id = await self.turns.user_speech_started()
        if resuming:
            self.transcripts.start_next_segment()
        else:
            self.transcripts.reset(turn_id)
        await self.stt.start_segment(turn_id, self.transcripts.segment_id)
        self._speech_started_at = asyncio.get_running_loop().time()
        self._speech_stopped_at = None
        self._first_partial_latency_ms = None
        self._segment_final_latency_ms = None
        self._turn_commit_latency_ms = None
        if self.turn_audio_buffer is None or self.turn_audio_buffer.turn_id != turn_id:
            self.turn_audio_buffer = TurnAudioBuffer(
                turn_id, maximum_seconds=self.config.audio.turn_buffer_seconds
            )
            self.turn_audio_buffer.extend(self.pre_roll_buffer.snapshot())
        return turn_id

    async def on_user_speech_stopped(self) -> None:
        """Record a VAD boundary; endpoint detection decides when to commit."""
        await self.turns.user_speech_stopped()
        self._speech_stopped_at = asyncio.get_running_loop().time()
        await self.stt.end_segment()
        await self._schedule_endpoint(self.state.current_turn)

    async def commit_turn(self, transcript: str) -> None:
        """Start response generation after endpoint detection accepts this turn."""
        await self._cancel_pending_endpoint()
        await self._commit_turn(transcript)

    async def wait_for_response(self) -> None:
        await self.turns.wait_for_response()

    def set_audio_frame_observer(
        self, observer: Callable[[AudioFrame], None] | None
    ) -> None:
        """Attach a lightweight asyncio-path probe; never called by PortAudio."""
        self._audio_frame_observer = observer

    @property
    def audio_connected(self) -> bool:
        return self._audio_task is not None and not self._audio_task.done()

    async def _run_audio_loop(self) -> None:
        assert self.audio_input is not None and self.vad is not None
        async for frame in self.audio_input.frames():
            if self._audio_frame_observer is not None:
                self._audio_frame_observer(frame)
            # Pre-roll is populated before VAD sees the frame, so a new turn
            # receives both lead-in audio and the triggering frame below.
            self.pre_roll_buffer.append(frame)
            was_speaking = self.state.turn_state is TurnState.USER_SPEAKING
            continuing_turn = self.state.turn_state in {
                TurnState.USER_SPEAKING,
                TurnState.AWAITING_COMMIT,
            }
            if continuing_turn and self.turn_audio_buffer:
                self.turn_audio_buffer.append(frame)
            self._audio_level = pcm16_rms_level(frame)
            vad_event = self.vad.process(frame)
            if vad_event and vad_event.kind is VADEventKind.SPEECH_STARTED:
                await self.on_user_speech_started()
                # Gate inference to speech, but prepend the bounded audio that
                # VAD needed in order to make its start decision.
                for pre_roll_frame in self.pre_roll_buffer.snapshot():
                    await self.stt.send_audio(pre_roll_frame.data)
            elif was_speaking:
                # send_audio is a bounded queue offer for real adapters; it
                # never performs network I/O on this runtime path.
                await self.stt.send_audio(frame.data)
            if vad_event and vad_event.kind is VADEventKind.SPEECH_STOPPED:
                await self.on_user_speech_stopped()
            await self._report_input_drops()

    async def _run_stt_loop(self) -> None:
        async for event in self.stt.events():
            if isinstance(event, STTServiceEvent):
                await self.events.publish(
                    ProviderStatusChanged(
                        provider=self.config.stt_provider,
                        status=event.status,
                        detail=event.detail,
                        connection_generation=event.connection_generation,
                    )
                )
                if event.status == "error":
                    await self.events.publish(
                        ProviderFailed(self.config.stt_provider, event.detail)
                    )
            else:
                await self._apply_stt_event(event)

    async def _apply_stt_event(self, event: STTEvent) -> None:
        now = asyncio.get_running_loop().time()
        self._last_stt_response_at = now
        turn_id = event.turn_id or self.state.current_turn
        if (
            turn_id is None
            or turn_id != self.state.current_turn
            or turn_id != self.transcripts.turn_id
            or (event.segment_id is not None and event.segment_id != self.transcripts.segment_id)
            or self.state.turn_state not in {TurnState.USER_SPEAKING, TurnState.AWAITING_COMMIT}
        ):
            return
        if not self.transcripts.apply(event):
            return
        if event.is_final:
            if self._speech_stopped_at is not None:
                self._segment_final_latency_ms = round(
                    (now - self._speech_stopped_at) * 1_000, 1
                )
            await self.events.publish(
                TranscriptFinal(turn_id, event.text, self.transcripts.segment_id)
            )
            if self.state.turn_state is TurnState.AWAITING_COMMIT and self.endpoint_state is not EndpointState.WAITING:
                await self._schedule_endpoint(turn_id)
        else:
            if self._first_partial_latency_ms is None and self._speech_started_at is not None:
                self._first_partial_latency_ms = round(
                    (now - self._speech_started_at) * 1_000, 1
                )
            await self.events.publish(
                TranscriptPartial(turn_id, event.text, self.transcripts.segment_id)
            )

    async def _schedule_endpoint(self, turn_id: str | None) -> None:
        if turn_id is None:
            return
        await self._cancel_pending_endpoint()
        self._endpoint_generation += 1
        generation = self._endpoint_generation
        segment_id = self.transcripts.segment_id
        self.endpoint_state = EndpointState.WAITING
        self._endpoint_deadline = (
            asyncio.get_running_loop().time()
            + self.config.endpoint.endpoint_timeout_ms / 1_000
        )
        self._endpoint_task = asyncio.create_task(
            self._endpoint_after_timeout(turn_id, segment_id, generation)
        )

    async def _cancel_pending_endpoint(self) -> None:
        self._endpoint_generation += 1
        task, self._endpoint_task = self._endpoint_task, None
        if task and task is not asyncio.current_task():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if self.endpoint_state is not EndpointState.COMMITTED:
            self.endpoint_state = EndpointState.IDLE
        self._endpoint_deadline = None

    async def _endpoint_after_timeout(
        self, turn_id: str, segment_id: int, generation: int
    ) -> None:
        try:
            await asyncio.sleep(self.config.endpoint.endpoint_timeout_ms / 1_000)
            if (
                generation != self._endpoint_generation
                or turn_id != self.state.current_turn
                or self.state.turn_state is not TurnState.AWAITING_COMMIT
                or turn_id != self.transcripts.turn_id
                or segment_id != self.transcripts.segment_id
            ):
                return
            transcript = self.transcripts.snapshot()
            if self.endpoint_detector.should_commit(transcript):
                await self._commit_turn(transcript.commit_text)
                self.endpoint_state = EndpointState.COMMITTED
            else:
                self.endpoint_state = EndpointState.IDLE
        finally:
            if self._endpoint_task is asyncio.current_task():
                self._endpoint_task = None
                self._endpoint_deadline = None

    async def _commit_turn(self, transcript: str) -> None:
        turn_id = self.state.current_turn
        await self.turns.commit_turn(transcript)
        await self.stt.commit_turn()
        self._last_committed_turn_id = turn_id
        if self._speech_stopped_at is not None:
            self._turn_commit_latency_ms = round(
                (asyncio.get_running_loop().time() - self._speech_stopped_at) * 1_000,
                1,
            )
        self._latency_history.append(
            {
                "turn_id": turn_id,
                "speech_start_to_first_partial_ms": self._first_partial_latency_ms,
                "speech_stop_to_segment_final_ms": self._segment_final_latency_ms,
                "speech_stop_to_commit_ms": self._turn_commit_latency_ms,
            }
        )
        logger.info("System 1 turn committed", extra={"turn_id": turn_id})

    async def _report_input_drops(self) -> None:
        if self.audio_input is None:
            return
        dropped = getattr(self.audio_input, "frames_dropped", 0)
        if dropped > self._reported_dropped_frames:
            self._reported_dropped_frames = dropped
            await self.events.publish(AudioFrameDropped(dropped))

    def debug_snapshot(self) -> dict[str, object]:
        """Small state only: debug tooling never receives raw microphone PCM."""
        transcript = self.transcripts.snapshot()
        provider = self.stt.diagnostics()
        now = asyncio.get_running_loop().time()
        endpoint_remaining_ms = (
            max(0.0, round((self._endpoint_deadline - now) * 1_000, 1))
            if self._endpoint_deadline is not None
            else None
        )
        latest_response_age_ms = (
            round((now - self._last_stt_response_at) * 1_000, 1)
            if self._last_stt_response_at is not None
            else provider.get("latest_response_age_ms")
        )
        return {
            "vad_state": self.vad.state.value if self.vad else VADState.LISTENING.value,
            "turn_state": self.state.turn_state.value,
            "audio_level": round(self._audio_level, 3),
            "vad_probability": self.vad.last_probability if self.vad else None,
            "vad_threshold": self.config.vad.speech_threshold,
            "frames_dropped": getattr(self.audio_input, "frames_dropped", 0),
            "audio_queue_size": getattr(self.audio_input, "queue_size", 0),
            "audio_device": self.config.audio.device,
            "audio_connected": self.audio_connected,
            "sample_rate": self.config.audio.sample_rate,
            "frame_duration_ms": self.config.audio.frame_duration_ms,
            "pre_roll_audio_seconds": round(self.pre_roll_buffer.duration_seconds, 3),
            "pre_roll_frame_count": self.pre_roll_buffer.frame_count,
            "buffered_audio_seconds": round(
                self.turn_audio_buffer.duration_seconds if self.turn_audio_buffer else 0.0, 3
            ),
            "current_turn_id": self.state.current_turn,
            "speech_segment_id": self.transcripts.segment_id,
            "partial_transcript": transcript.partial,
            "final_transcript": transcript.final,
            "best_transcript": transcript.best,
            "latest_segment_final": transcript.latest_segment_final,
            "endpoint_state": self.endpoint_state.value,
            "endpoint_pending": self._endpoint_task is not None and not self._endpoint_task.done(),
            "endpoint_remaining_ms": endpoint_remaining_ms,
            "turn_committed": self._last_committed_turn_id == self.state.current_turn,
            "stt": provider,
            "latency": {
                "speech_start_to_first_partial_ms": self._first_partial_latency_ms,
                "speech_stop_to_segment_final_ms": self._segment_final_latency_ms,
                "speech_stop_to_commit_ms": self._turn_commit_latency_ms,
                "latest_stt_response_age_ms": latest_response_age_ms,
                "history": list(self._latency_history),
            },
        }


def default_provider_registry(config: System1Config | None = None) -> ProviderRegistry:
    """Development-only composition; production registers adapter factories here."""
    from .adapters.nemotron_stt import NemotronSTTProvider

    config = config or System1Config()
    registry = ProviderRegistry()
    registry.register("mock.stt", MockSTTProvider)
    registry.register("mock", MockSTTProvider)
    registry.register("nemotron", lambda: NemotronSTTProvider(config.nemotron_stt))
    registry.register("mock.dialogue", MockDialogueProvider)
    registry.register("mock.tts", MockTTSProvider)
    return registry
