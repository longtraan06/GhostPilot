"""The local-first barge-in path."""

from __future__ import annotations

import asyncio

from .event_bus import EventBus
from .events import ConversationInterrupted, GenerationCancelled
from .providers import DialogueCancellationHandle, DialogueProvider, Playback, TTSProvider
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
        # Capture old request ownership before the background cleanup can run.
        dialogue_cleanup = self._dialogue.invalidate_active()
        await self._events.publish(ConversationInterrupted(interrupted_turn))
        task = asyncio.create_task(self._cancel_providers(interrupted_turn, dialogue_cleanup))
        self._cancellation_tasks.add(task)
        task.add_done_callback(self._cancellation_tasks.discard)

    async def _cancel_providers(
        self,
        interrupted_turn: str | None,
        dialogue_cleanup: DialogueCancellationHandle | None,
    ) -> None:
        try:
            cleanup = dialogue_cleanup.close() if dialogue_cleanup is not None else _noop()
            # TODO(M5): TTS needs request-specific cancellation ownership too.
            await asyncio.gather(self._tts.cancel(), cleanup, return_exceptions=True)
        finally:
            await self._events.publish(GenerationCancelled(interrupted_turn))

    async def close(self) -> None:
        """Join captured cleanup so invalidated transports are actually closed."""
        tasks = tuple(self._cancellation_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_for_cancellations(self) -> None:
        """Join currently scheduled cleanup; useful for deterministic lifecycle tests."""
        tasks = tuple(self._cancellation_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def _noop() -> None:
    return None
