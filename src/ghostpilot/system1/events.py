"""Typed, transport-independent events emitted by System 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class AudioSpeechStarted:
    turn_id: str
    name: Literal["audio.speech_started"] = "audio.speech_started"


@dataclass(frozen=True, slots=True)
class AudioSpeechStopped:
    turn_id: str
    name: Literal["audio.speech_stopped"] = "audio.speech_stopped"


@dataclass(frozen=True, slots=True)
class AudioInputStarted:
    name: Literal["audio.input_started"] = "audio.input_started"


@dataclass(frozen=True, slots=True)
class AudioInputStopped:
    name: Literal["audio.input_stopped"] = "audio.input_stopped"


@dataclass(frozen=True, slots=True)
class AudioFrameDropped:
    dropped_frames: int
    name: Literal["audio.frame_dropped"] = "audio.frame_dropped"


@dataclass(frozen=True, slots=True)
class TranscriptPartial:
    turn_id: str
    text: str
    segment_id: int | None = None
    name: Literal["transcript.partial"] = "transcript.partial"


@dataclass(frozen=True, slots=True)
class TranscriptFinal:
    turn_id: str
    text: str
    segment_id: int | None = None
    name: Literal["transcript.final"] = "transcript.final"


@dataclass(frozen=True, slots=True)
class ConversationTurnStarted:
    turn_id: str
    name: Literal["conversation.turn_started"] = "conversation.turn_started"


@dataclass(frozen=True, slots=True)
class ConversationTurnCommitted:
    turn_id: str
    transcript: str
    name: Literal["conversation.turn_committed"] = "conversation.turn_committed"


@dataclass(frozen=True, slots=True)
class ConversationAssistantSpeaking:
    turn_id: str
    name: Literal["conversation.assistant_speaking"] = "conversation.assistant_speaking"


@dataclass(frozen=True, slots=True)
class ConversationInterrupted:
    interrupted_turn_id: str | None
    name: Literal["conversation.interrupted"] = "conversation.interrupted"


@dataclass(frozen=True, slots=True)
class GenerationStarted:
    turn_id: str
    name: Literal["generation.started"] = "generation.started"


@dataclass(frozen=True, slots=True)
class GenerationCancelled:
    turn_id: str | None
    name: Literal["generation.cancelled"] = "generation.cancelled"


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    turn_id: str
    text: str
    name: Literal["speech.started"] = "speech.started"


@dataclass(frozen=True, slots=True)
class SpeechFinished:
    turn_id: str
    name: Literal["speech.finished"] = "speech.finished"


@dataclass(frozen=True, slots=True)
class DialogueActionProposed:
    turn_id: str
    action: dict[str, Any]
    name: Literal["dialogue.action_proposed"] = "dialogue.action_proposed"


@dataclass(frozen=True, slots=True)
class ProviderFailed:
    provider: str
    detail: str
    name: Literal["system.provider_failed"] = "system.provider_failed"


@dataclass(frozen=True, slots=True)
class ProviderStatusChanged:
    provider: str
    status: str
    detail: str = ""
    connection_generation: int = 0
    name: Literal["system.provider_status"] = "system.provider_status"


DomainEvent: TypeAlias = (
    AudioSpeechStarted
    | AudioSpeechStopped
    | AudioInputStarted
    | AudioInputStopped
    | AudioFrameDropped
    | TranscriptPartial
    | TranscriptFinal
    | ConversationTurnStarted
    | ConversationTurnCommitted
    | ConversationAssistantSpeaking
    | ConversationInterrupted
    | GenerationStarted
    | GenerationCancelled
    | SpeechStarted
    | SpeechFinished
    | DialogueActionProposed
    | ProviderFailed
    | ProviderStatusChanged
)
