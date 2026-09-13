"""The local-first barge-in path."""

from __future__ import annotations

import asyncio

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
        self._cancellation_tasks: set[asyncio.Task[None]] = set()

    async def interrupt(self, next_turn_id: str) -> None:
        """Claim the new user turn without awaiting remote provider teardown."""
        self._playback.stop_now()
        interrupted_turn = self._state.mark_interrupted()
        # The new user owns the conversation before a cloud cancellation returns.
        self._state.begin_user_turn(next_turn_id)
        await self._events.publish(ConversationInterrupted(interrupted_turn))
        task = asyncio.create_task(self._cancel_providers(interrupted_turn))
        self._cancellation_tasks.add(task)
        task.add_done_callback(self._cancellation_tasks.discard)

    async def _cancel_providers(self, interrupted_turn: str | None) -> None:
        try:
            await asyncio.gather(
                self._tts.cancel(), self._dialogue.cancel(), return_exceptions=True
            )
        finally:
            await self._events.publish(GenerationCancelled(interrupted_turn))

    async def close(self) -> None:
        """Cancel and join managed background cleanup during runtime shutdown."""
        tasks = tuple(self._cancellation_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_for_cancellations(self) -> None:
        """Join currently scheduled cleanup; useful for deterministic lifecycle tests."""
        tasks = tuple(self._cancellation_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
