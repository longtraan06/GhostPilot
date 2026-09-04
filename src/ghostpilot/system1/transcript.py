"""Turn-scoped transcript state, independent from VAD and STT providers."""

from __future__ import annotations

from dataclasses import dataclass

from .providers import STTEvent


@dataclass(frozen=True, slots=True)
class TranscriptSnapshot:
    turn_id: str | None
    partial: str
    final: str

    @property
    def best(self) -> str:
        return self.final or self.partial


class TranscriptManager:
    def __init__(self) -> None:
        self._turn_id: str | None = None
        self._partial = ""
        self._final = ""

    @property
    def turn_id(self) -> str | None:
        return self._turn_id

    def reset(self, turn_id: str) -> None:
        self._turn_id = turn_id
        self._partial = ""
        self._final = ""

    def apply(self, event: STTEvent) -> bool:
        """Apply snapshot-style STT updates without concatenating overlap."""
        if self._turn_id is None:
            return False
        if event.is_final:
            if event.text == self._final:
                return False
            self._final = event.text
            return True
        if event.text == self._partial:
            return False
        self._partial = event.text
        return True

    def snapshot(self) -> TranscriptSnapshot:
        return TranscriptSnapshot(self._turn_id, self._partial, self._final)
