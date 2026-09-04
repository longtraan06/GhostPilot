"""Small latency-bounded text segmenter for streaming TTS input."""

from __future__ import annotations


class SpeechSegmenter:
    def __init__(self, *, max_characters: int = 140) -> None:
        self.max_characters = max_characters
        self._buffer = ""

    def push(self, text: str) -> list[str]:
        self._buffer += text
        boundaries = ".?!,;:\n"
        emitted: list[str] = []
        while self._buffer:
            candidates = [self._buffer.rfind(mark) for mark in boundaries]
            boundary = max(candidates)
            if boundary >= 0:
                emitted.append(self._buffer[: boundary + 1].strip())
                self._buffer = self._buffer[boundary + 1 :].lstrip()
            elif len(self._buffer) >= self.max_characters:
                split = self._buffer.rfind(" ", 0, self.max_characters)
                split = split if split > 0 else self.max_characters
                emitted.append(self._buffer[:split].strip())
                self._buffer = self._buffer[split:].lstrip()
            else:
                break
        return [segment for segment in emitted if segment]

    def flush(self) -> str | None:
        segment, self._buffer = self._buffer.strip(), ""
        return segment or None
