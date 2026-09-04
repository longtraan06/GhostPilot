"""Configuration and provider registry for swapping adapters at composition time."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar


@dataclass(frozen=True, slots=True)
class System1Config:
    stt_provider: str = "mock.stt"
    dialogue_provider: str = "mock.dialogue"
    tts_provider: str = "mock.tts"


T = TypeVar("T")


class ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], object]] = {}

    def register(self, name: str, factory: Callable[[], T]) -> None:
        self._factories[name] = factory

    def build(self, name: str) -> T:
        try:
            return self._factories[name]()  # type: ignore[return-value]
        except KeyError as error:
            raise ValueError(f"unknown provider: {name}") from error
