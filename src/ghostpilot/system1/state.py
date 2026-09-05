"""Conversation state and the explicit System 1 turn state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UserState(str, Enum):
    IDLE = "IDLE"
    SPEAKING = "SPEAKING"


class AssistantState(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class TurnState(str, Enum):
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    AWAITING_COMMIT = "AWAITING_COMMIT"
    THINKING = "THINKING"
    ASSISTANT_SPEAKING = "ASSISTANT_SPEAKING"
    INTERRUPTED = "INTERRUPTED"


@dataclass(slots=True)
class ConversationState:
    """State owned by System 1, never application/task state."""

    user_state: UserState = UserState.IDLE
    assistant_state: AssistantState = AssistantState.IDLE
    turn_state: TurnState = TurnState.LISTENING
    current_turn: str | None = None
    active_generation: str | None = None
    active_speech: str | None = None
    partial_transcript: str = ""
    committed_transcript: str = ""

    def begin_user_turn(self, turn_id: str) -> None:
        self.current_turn = turn_id
        self.user_state = UserState.SPEAKING
        self.assistant_state = AssistantState.IDLE
        self.active_generation = None
        self.active_speech = None
        self.partial_transcript = ""
        self.turn_state = TurnState.USER_SPEAKING

    def commit_turn(self, transcript: str) -> None:
        if self.turn_state is not TurnState.AWAITING_COMMIT:
            raise RuntimeError("cannot commit until endpoint detection commits the stopped turn")
        self.user_state = UserState.IDLE
        self.committed_transcript = transcript
        self.assistant_state = AssistantState.THINKING
        self.active_generation = self.current_turn
        self.turn_state = TurnState.THINKING

    def stop_user_speech(self) -> None:
        if self.user_state is not UserState.SPEAKING:
            raise RuntimeError("cannot stop speech without an active user turn")
        self.user_state = UserState.IDLE
        self.turn_state = TurnState.AWAITING_COMMIT

    def resume_user_speech(self) -> None:
        if self.turn_state is not TurnState.AWAITING_COMMIT:
            raise RuntimeError("can only resume a user turn awaiting endpoint detection")
        self.user_state = UserState.SPEAKING
        self.turn_state = TurnState.USER_SPEAKING

    def abort_user_turn(self) -> str:
        """Invalidate an uncommitted user turn without pretending it committed."""
        if self.turn_state not in {TurnState.USER_SPEAKING, TurnState.AWAITING_COMMIT}:
            raise RuntimeError("can only abort an active uncommitted user turn")
        if self.current_turn is None:
            raise RuntimeError("an active user turn requires a current turn")
        aborted_turn = self.current_turn
        self.current_turn = None
        self.user_state = UserState.IDLE
        self.assistant_state = AssistantState.IDLE
        self.active_generation = None
        self.active_speech = None
        self.partial_transcript = ""
        self.turn_state = TurnState.LISTENING
        return aborted_turn

    def begin_assistant_speech(self) -> None:
        if self.assistant_state is not AssistantState.THINKING:
            raise RuntimeError("assistant may speak only after thinking")
        self.assistant_state = AssistantState.SPEAKING
        self.active_speech = self.current_turn
        self.turn_state = TurnState.ASSISTANT_SPEAKING

    def finish_assistant_turn(self) -> None:
        self.assistant_state = AssistantState.IDLE
        self.active_generation = None
        self.active_speech = None
        self.turn_state = TurnState.LISTENING

    def mark_interrupted(self) -> str | None:
        interrupted = self.current_turn
        self.assistant_state = AssistantState.INTERRUPTED
        self.active_generation = None
        self.active_speech = None
        self.turn_state = TurnState.INTERRUPTED
        return interrupted
