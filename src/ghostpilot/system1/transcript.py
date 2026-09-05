"""Turn-scoped transcript state, independent from VAD and STT providers."""

from __future__ import annotations

from dataclasses import dataclass

from .providers import STTEvent


@dataclass(frozen=True, slots=True)
class TranscriptSnapshot:
    turn_id: str | None
    partial: str
    final: str
    segment_id: int
    latest_segment_final: bool
    latest_segment_text: str

    @property
    def best(self) -> str:
        """Most recent snapshot text, irrespective of endpoint eligibility."""
        return self.latest_segment_text or self.final or self.partial

    @property
    def commit_text(self) -> str:
        """Text eligible for endpoint commit in the active speech segment."""
        return self.latest_segment_text if self.latest_segment_final else ""


class TranscriptManager:
    def __init__(self) -> None:
        self._turn_id: str | None = None
        self._partial = ""
        self._final = ""
        self._segment_id = 0
        self._latest_segment_final = False
        self._latest_segment_text = ""

    @property
    def turn_id(self) -> str | None:
        return self._turn_id

    @property
    def segment_id(self) -> int:
        return self._segment_id

    def reset(self, turn_id: str) -> None:
        self._turn_id = turn_id
        self._partial = ""
        self._final = ""
        self._segment_id = 1
        self._latest_segment_final = False
        self._latest_segment_text = ""

    def clear(self) -> None:
        """Forget an invalidated turn and reject all delayed provider results."""
        self._turn_id = None
        self._partial = ""
        self._final = ""
        self._segment_id = 0
        self._latest_segment_final = False
        self._latest_segment_text = ""

    def start_next_segment(self) -> int:
        """Begin another VAD speech segment without discarding turn history."""
        if self._turn_id is None:
            raise RuntimeError("cannot resume a transcript without an active turn")
        self._segment_id += 1
        self._latest_segment_final = False
        self._latest_segment_text = ""
        return self._segment_id

    def apply(self, event: STTEvent) -> bool:
        """Apply snapshot-style STT updates without concatenating overlap."""
        if self._turn_id is None:
            return False
        if event.segment_id is not None and event.segment_id != self._segment_id:
            return False
        if event.is_final:
            changed = (
                event.text != self._final
                or event.text != self._latest_segment_text
                or not self._latest_segment_final
            )
            self._final = event.text
            self._latest_segment_text = event.text
            self._latest_segment_final = True
            return changed
        changed = event.text != self._partial or self._latest_segment_final
        self._partial = event.text
        self._latest_segment_text = event.text
        # A partial is a newer, not-yet-stable revision of this segment.
        self._latest_segment_final = False
        return changed

    def snapshot(self) -> TranscriptSnapshot:
        return TranscriptSnapshot(
            self._turn_id,
            self._partial,
            self._final,
            self._segment_id,
            self._latest_segment_final,
            self._latest_segment_text,
        )
