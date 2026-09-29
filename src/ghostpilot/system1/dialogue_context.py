"""Bounded, provider-neutral short-term dialogue context for System 1."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config import DialogueContextConfig
from .providers import DialogueMessage


@dataclass(frozen=True, slots=True)
class DialogueExchange:
    user: str
    assistant: str

    @property
    def character_count(self) -> int:
        return len(self.user) + len(self.assistant)


class ConversationHistory:
    """In-memory canonical exchanges; only TurnManager decides when to append."""

    def __init__(self, config: DialogueContextConfig | None = None) -> None:
        self._exchanges: list[DialogueExchange] = []
        self._config = config or DialogueContextConfig()

    def append_exchange(self, user_text: str, assistant_text: str) -> None:
        if not user_text or not assistant_text:
            raise ValueError("completed dialogue exchanges require user and assistant text")
        self._exchanges.append(DialogueExchange(user_text, assistant_text))
        # Retain only exchanges that can still contribute to future context.
        # This keeps a long-running voice session bounded in both count and text.
        self._exchanges = list(_select_exchanges(self._exchanges, self._config))

    def exchanges(self) -> tuple[DialogueExchange, ...]:
        return tuple(self._exchanges)

    def clear(self) -> None:
        self._exchanges.clear()

    @property
    def exchange_count(self) -> int:
        return len(self._exchanges)

    @property
    def message_count(self) -> int:
        return self.exchange_count * 2

    @property
    def character_count(self) -> int:
        return sum(exchange.character_count for exchange in self._exchanges)


@dataclass(frozen=True, slots=True)
class BuiltDialogueContext:
    messages: tuple[DialogueMessage, ...]
    historical_exchanges: int
    historical_messages: int
    historical_chars: int
    total_messages: int
    total_chars: int
    estimated_tokens: int


class DialogueContextBuilder:
    """Build a bounded prompt from complete past exchanges and one current user turn."""

    def __init__(self, config: DialogueContextConfig) -> None:
        self.config = config

    def build(
        self, history: ConversationHistory, current_user_text: str
    ) -> BuiltDialogueContext:
        selected = _select_exchanges(history.exchanges(), self.config)
        historical_chars = sum(exchange.character_count for exchange in selected)
        messages: list[DialogueMessage] = [
            DialogueMessage("system", self.config.system_prompt)
        ]
        for exchange in selected:
            messages.extend(
                (
                    DialogueMessage("user", exchange.user),
                    DialogueMessage("assistant", exchange.assistant),
                )
            )
        messages.append(DialogueMessage("user", current_user_text))
        immutable_messages = tuple(messages)
        total_chars = sum(len(message.content) for message in immutable_messages)
        return BuiltDialogueContext(
            messages=immutable_messages,
            historical_exchanges=len(selected),
            historical_messages=len(selected) * 2,
            historical_chars=historical_chars,
            total_messages=len(immutable_messages),
            total_chars=total_chars,
            estimated_tokens=math.ceil(total_chars / 4),
        )


def _select_exchanges(
    exchanges: tuple[DialogueExchange, ...] | list[DialogueExchange],
    config: DialogueContextConfig,
) -> tuple[DialogueExchange, ...]:
    """Select newest eligible complete exchanges, then restore chronology."""
    selected_reversed: list[DialogueExchange] = []
    historical_chars = 0
    for exchange in reversed(exchanges):
        if len(selected_reversed) >= config.max_history_exchanges:
            break
        cost = exchange.character_count
        if historical_chars + cost > config.max_history_chars:
            continue
        selected_reversed.append(exchange)
        historical_chars += cost
    return tuple(reversed(selected_reversed))
