"""The local-first barge-in path."""

from __future__ import annotations

from .event_bus import EventBus
from .events import ConversationInterrupted, GenerationCancelled
from .providers import DialogueProvider, Playback, TTSProvider
from .state import ConversationState


class InterruptionController:
    def __init__(
        self,
        state: ConversationState,
        events: EventBus,
        dialogue: DialogueProvider,
        tts: TTSProvider,
        playback: Playback,
    ) -> None:
        self._state, self._events = state, events
        self._dialogue, self._tts, self._playback = dialogue, tts, playback

    async def interrupt(self, next_turn_id: str) -> None:
        """Stop local audio synchronously before any awaited provider cancellation."""
        self._playback.stop_now()
        interrupted_turn = self._state.mark_interrupted()
        # The new user owns the conversation before a cloud cancellation returns.
        self._state.begin_user_turn(next_turn_id)
        await self._events.publish(ConversationInterrupted(interrupted_turn))
        await self._tts.cancel()
        await self._dialogue.cancel()
        await self._events.publish(GenerationCancelled(interrupted_turn))
