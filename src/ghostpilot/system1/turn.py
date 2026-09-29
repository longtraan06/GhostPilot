"""Turn orchestration without any vendor-specific logic."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from .event_bus import EventBus
from .dialogue_context import (
    BuiltDialogueContext,
    ConversationHistory,
    DialogueContextBuilder,
)
from .events import (
    AudioSpeechStarted,
    AudioSpeechStopped,
    ConversationAssistantSpeaking,
    ConversationTurnAborted,
    ConversationTurnCommitted,
    ConversationTurnStarted,
    DialogueActionProposed,
    GenerationStarted,
    ProviderFailed,
    SpeechFinished,
    SpeechStarted,
)
from .interruption import InterruptionController
from .providers import DialogueProvider, DialogueProviderError, Playback, TTSProvider
from .speech import SpeechSegmenter
from .state import AssistantState, ConversationState, TurnState


class TurnManager:
    def __init__(
        self,
        state: ConversationState,
        events: EventBus,
        dialogue: DialogueProvider,
        tts: TTSProvider,
        playback: Playback,
        interruption: InterruptionController,
        conversation_history: ConversationHistory,
        context_builder: DialogueContextBuilder,
    ) -> None:
        self.state, self.events = state, events
        self._dialogue, self._tts, self._playback = dialogue, tts, playback
        self._interruption = interruption
        self.history = conversation_history
        self._context_builder = context_builder
        self._last_context: BuiltDialogueContext | None = None
        self._turn_number = 0
        self._response_task: asyncio.Task[None] | None = None
        self._response_tasks: set[asyncio.Task[None]] = set()

    async def user_speech_started(self) -> str:
        if self.state.turn_state is TurnState.AWAITING_COMMIT:
            turn_id = self.state.current_turn
            if turn_id is None:
                raise RuntimeError("an endpoint-waiting state requires a current turn")
            self.state.resume_user_speech()
            await self.events.publish(AudioSpeechStarted(turn_id))
            return turn_id

        self._turn_number += 1
        turn_id = f"turn-{self._turn_number}"
        if self.state.assistant_state in {AssistantState.THINKING, AssistantState.SPEAKING}:
            await self._interruption.interrupt(turn_id)
        else:
            self.state.begin_user_turn(turn_id)
        await self.events.publish(AudioSpeechStarted(turn_id))
        await self.events.publish(ConversationTurnStarted(turn_id))
        return turn_id

    async def user_speech_stopped(self) -> None:
        turn_id = self.state.current_turn
        if turn_id is None:
            raise RuntimeError("cannot stop speech without a turn")
        await self.events.publish(AudioSpeechStopped(turn_id))
        self.state.stop_user_speech()

    async def commit_turn(self, transcript: str) -> None:
        turn_id = self.state.current_turn
        if turn_id is None:
            raise RuntimeError("cannot commit without a turn")
        self.state.commit_turn(transcript)
        await self.events.publish(ConversationTurnCommitted(turn_id, transcript))
        task = asyncio.create_task(self._respond(turn_id, transcript))
        self._response_task = task
        self._response_tasks.add(task)
        task.add_done_callback(self._response_tasks.discard)

    async def abort_user_turn(self, reason: str, connection_generation: int = 0) -> str:
        """Return safely to listening without committing or starting dialogue."""
        turn_id = self.state.abort_user_turn()
        await self.events.publish(
            ConversationTurnAborted(turn_id, reason, connection_generation)
        )
        return turn_id

    async def wait_for_response(self) -> None:
        if self._response_task:
            await self._response_task

    async def close(self) -> None:
        """End every response task and provider cleanup owned by this manager."""
        self._playback.stop_now()
        await self._interruption.close()
        await asyncio.gather(
            self._dialogue.cancel(), self._tts.cancel(), return_exceptions=True
        )
        tasks = tuple(self._response_tasks)
        self._response_task = None
        for task in tasks:
            if task is not asyncio.current_task():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _respond(self, turn_id: str, transcript: str) -> None:
        await self.events.publish(GenerationStarted(turn_id))
        segmenter = SpeechSegmenter()
        context = self._context_builder.build(self.history, transcript)
        self._last_context = context
        assistant_parts: list[str] = []
        completed = False
        try:
            async for output in self._dialogue.stream(context.messages):
                if output.text:
                    assistant_parts.append(output.text)
                if output.action:
                    await self.events.publish(DialogueActionProposed(turn_id, output.action))
                for segment in segmenter.push(output.text):
                    await self._speak(turn_id, segment)
            if (segment := segmenter.flush()) is not None:
                await self._speak(turn_id, segment)
            completed = True
        except DialogueProviderError as error:
            await self.events.publish(ProviderFailed(self._dialogue_provider_name(), str(error)))
        finally:
            assistant_text = "".join(assistant_parts)
            if (
                completed
                and assistant_text
                and self.state.current_turn == turn_id
                and self.state.assistant_state is not AssistantState.INTERRUPTED
            ):
                # This synchronous ownership check prevents an old completed stream
                # from canonicalising history after a newer user turn takes over.
                self.history.append_exchange(transcript, assistant_text)
            # A newer user turn owns the state after barge-in.
            if self.state.current_turn == turn_id and self.state.assistant_state is not AssistantState.INTERRUPTED:
                self.state.finish_assistant_turn()

    def clear_dialogue_history(self) -> None:
        self.history.clear()
        self._last_context = None

    def context_diagnostics(self) -> dict[str, int]:
        context = self._last_context
        return {
            "completed_exchanges": self.history.exchange_count,
            "history_messages": self.history.message_count,
            "history_chars": self.history.character_count,
            "max_history_exchanges": self._context_builder.config.max_history_exchanges,
            "max_history_chars": self._context_builder.config.max_history_chars,
            "last_context_messages": context.total_messages if context else 0,
            "last_context_chars": context.total_chars if context else 0,
            "estimated_tokens": context.estimated_tokens if context else 0,
        }

    def _dialogue_provider_name(self) -> str:
        diagnostics = getattr(self._dialogue, "diagnostics", None)
        if callable(diagnostics):
            provider = diagnostics().get("provider")
            if isinstance(provider, str) and provider:
                return provider
        return type(self._dialogue).__name__

    async def _speak(self, turn_id: str, text: str) -> None:
        if self.state.current_turn != turn_id or self._dialogue_cancelled():
            return
        if self.state.assistant_state is AssistantState.THINKING:
            self.state.begin_assistant_speech()
            await self.events.publish(ConversationAssistantSpeaking(turn_id))
        await self.events.publish(SpeechStarted(turn_id, text))
        async for audio in self._tts.stream(text):
            if self.state.current_turn != turn_id or self._dialogue_cancelled():
                return
            await self._playback.play(audio)
        await self.events.publish(SpeechFinished(turn_id))

    def _dialogue_cancelled(self) -> bool:
        # Providers deliberately share no vendor API here; cancellation changes state first.
        return self.state.assistant_state is AssistantState.INTERRUPTED
