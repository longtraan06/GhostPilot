"""Turn orchestration without any vendor-specific logic."""

from __future__ import annotations

import asyncio

from .event_bus import EventBus
from .events import (
    AudioSpeechStarted,
    AudioSpeechStopped,
    ConversationAssistantSpeaking,
    ConversationTurnCommitted,
    ConversationTurnStarted,
    DialogueActionProposed,
    GenerationStarted,
    SpeechFinished,
    SpeechStarted,
)
from .interruption import InterruptionController
from .providers import DialogueProvider, Playback, TTSProvider
from .speech import SpeechSegmenter
from .state import AssistantState, ConversationState


class TurnManager:
    def __init__(
        self,
        state: ConversationState,
        events: EventBus,
        dialogue: DialogueProvider,
        tts: TTSProvider,
        playback: Playback,
        interruption: InterruptionController,
    ) -> None:
        self.state, self.events = state, events
        self._dialogue, self._tts, self._playback = dialogue, tts, playback
        self._interruption = interruption
        self._turn_number = 0
        self._response_task: asyncio.Task[None] | None = None

    async def user_speech_started(self) -> str:
        self._turn_number += 1
        turn_id = f"turn-{self._turn_number}"
        if self.state.assistant_state in {AssistantState.THINKING, AssistantState.SPEAKING}:
            await self._interruption.interrupt(turn_id)
        else:
            self.state.begin_user_turn(turn_id)
        await self.events.publish(AudioSpeechStarted(turn_id))
        await self.events.publish(ConversationTurnStarted(turn_id))
        return turn_id

    async def user_speech_stopped(self, transcript: str) -> None:
        turn_id = self.state.current_turn
        if turn_id is None:
            raise RuntimeError("cannot stop speech without a turn")
        await self.events.publish(AudioSpeechStopped(turn_id))
        self.state.commit_turn(transcript)
        await self.events.publish(ConversationTurnCommitted(turn_id, transcript))
        self._response_task = asyncio.create_task(self._respond(turn_id, transcript))

    async def wait_for_response(self) -> None:
        if self._response_task:
            await self._response_task

    async def _respond(self, turn_id: str, transcript: str) -> None:
        await self.events.publish(GenerationStarted(turn_id))
        segmenter = SpeechSegmenter()
        try:
            async for output in self._dialogue.stream(transcript):
                if output.action:
                    await self.events.publish(DialogueActionProposed(turn_id, output.action))
                for segment in segmenter.push(output.text):
                    await self._speak(turn_id, segment)
            if (segment := segmenter.flush()) is not None:
                await self._speak(turn_id, segment)
        finally:
            # A newer user turn owns the state after barge-in.
            if self.state.current_turn == turn_id and self.state.assistant_state is not AssistantState.INTERRUPTED:
                self.state.finish_assistant_turn()

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
