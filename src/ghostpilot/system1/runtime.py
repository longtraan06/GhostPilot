"""Minimal System 1 composition root."""

from __future__ import annotations

from typing import cast

from .config import ProviderRegistry, System1Config
from .event_bus import EventBus
from .interruption import InterruptionController
from .mock_providers import MockDialogueProvider, MockPlayback, MockSTTProvider, MockTTSProvider
from .providers import DialogueProvider, Playback, STTProvider, TTSProvider
from .state import ConversationState
from .turn import TurnManager


class System1Runtime:
    def __init__(
        self,
        *,
        stt: STTProvider | None = None,
        dialogue: DialogueProvider | None = None,
        tts: TTSProvider | None = None,
        playback: Playback | None = None,
    ) -> None:
        self.events = EventBus()
        self.state = ConversationState()
        self.stt = stt or MockSTTProvider()
        self.dialogue = dialogue or MockDialogueProvider()
        self.tts = tts or MockTTSProvider()
        self.playback = playback or MockPlayback()
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
        )

    async def start(self) -> None:
        await self.stt.connect()

    async def close(self) -> None:
        await self.stt.close()

    async def on_user_speech_started(self) -> str:
        return await self.turns.user_speech_started()

    async def on_user_speech_stopped(self, transcript: str) -> None:
        await self.turns.user_speech_stopped(transcript)

    async def wait_for_response(self) -> None:
        await self.turns.wait_for_response()


def default_provider_registry() -> ProviderRegistry:
    """Development-only composition; production registers adapter factories here."""
    registry = ProviderRegistry()
    registry.register("mock.stt", MockSTTProvider)
    registry.register("mock.dialogue", MockDialogueProvider)
    registry.register("mock.tts", MockTTSProvider)
    return registry
