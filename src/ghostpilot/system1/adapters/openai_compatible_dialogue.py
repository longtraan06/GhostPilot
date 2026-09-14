"""OpenAI-compatible streaming dialogue adapter for System 1.

The OpenAI SDK stays in this adapter.  System 1 receives only its small,
provider-neutral ``DialogueOutput`` stream.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import inspect
import logging
from typing import Any

try:  # Keep mock-only development and unit tests usable before the extra is installed.
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - exercised by composition, not the fake-client tests
    AsyncOpenAI = None  # type: ignore[assignment,misc]

from ..config import OpenAICompatibleDialogueConfig
from ..providers import DialogueCancellationHandle, DialogueOutput, DialogueProviderError


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _OpenAIDialogueCancellation:
    """Captured cleanup for one invalidated request, never a future request."""

    provider: "OpenAICompatibleDialogueProvider"
    generation: int
    stream: Any | None
    closed: bool = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        started_at = asyncio.get_running_loop().time()
        await _close_stream(self.stream)
        self.provider._last_cancellation_latency_ms = round(
            (asyncio.get_running_loop().time() - started_at) * 1_000, 1
        )


def _normalise_base_url(base_url: str) -> str:
    """Return the OpenAI API root while accepting the vLLM server root."""
    normalised = base_url.strip().rstrip("/")
    return normalised if normalised.endswith("/v1") else f"{normalised}/v1"


class OpenAICompatibleDialogueProvider:
    """Low-latency, cancellation-safe Chat Completions streaming provider."""

    def __init__(
        self,
        config: OpenAICompatibleDialogueConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self._config = config
        self._base_url = _normalise_base_url(config.base_url)
        if client is None:
            if AsyncOpenAI is None:
                raise RuntimeError("Install dialogue support: pip install -e '.[dialogue]'")
            client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=config.api_key,
                timeout=config.timeout_seconds,
            )
        self._client = client
        self._generation = 0
        self._active_generation: int | None = None
        self._active_stream: Any | None = None
        self._active_started_at: float | None = None
        self._requests_started = 0
        self._requests_completed = 0
        self._requests_cancelled = 0
        self._requests_failed = 0
        self._chunks_received = 0
        self._characters_received = 0
        self._last_error = ""
        self._last_finish_reason = ""
        self._last_ttft_ms: float | None = None
        self._last_total_duration_ms: float | None = None
        self._last_cancellation_latency_ms: float | None = None
        self._live_text = ""

    async def stream(self, transcript: str) -> AsyncIterator[DialogueOutput]:
        """Yield content deltas immediately while this request owns the generation."""
        loop = asyncio.get_running_loop()
        self._generation += 1
        generation = self._generation
        started_at = loop.time()
        self._active_generation = generation
        self._active_stream = None
        self._active_started_at = started_at
        self._requests_started += 1
        self._last_error = ""
        self._last_finish_reason = ""
        self._last_ttft_ms = None
        self._last_total_duration_ms = None
        self._live_text = ""
        request_stream: Any | None = None
        first_content = True
        completed = False

        try:
            if not self._config.model.strip():
                raise ValueError(
                    "GHOSTPILOT_DIALOGUE_MODEL is required for the openai-compatible provider"
                )
            request_options: dict[str, Any] = {
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": self._config.system_prompt},
                    {"role": "user", "content": transcript},
                ],
                "stream": True,
                "temperature": self._config.temperature,
                "max_tokens": self._config.max_tokens,
            }
            # OpenAI SDK's extra_body preserves compatibility while allowing a
            # server-specific extension such as llama.cpp chat template options.
            if self._config.extra_body:
                request_options["extra_body"] = self._config.extra_body
            request_stream = await self._client.chat.completions.create(
                **request_options,
            )
            if generation != self._active_generation:
                return
            self._active_stream = request_stream

            async for chunk in request_stream:
                if generation != self._active_generation:
                    return
                content, finish_reason = _chunk_content_and_finish_reason(chunk)
                if finish_reason:
                    self._last_finish_reason = finish_reason
                if not content:
                    continue
                now = loop.time()
                if first_content:
                    first_content = False
                    self._last_ttft_ms = round((now - started_at) * 1_000, 1)
                self._chunks_received += 1
                self._characters_received += len(content)
                self._live_text += content
                # Check ownership immediately before yielding too: consumer code may
                # start another turn between receiving the network chunk and this point.
                if generation == self._active_generation:
                    yield DialogueOutput(text=content)
            completed = True
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Preserve diagnostics, then let TurnManager surface a provider-neutral
            # failure event and recover the conversation state.
            if generation == self._active_generation:
                self._requests_failed += 1
                self._last_error = f"{type(error).__name__}: {error}"
                logger.warning("Dialogue request failed: %s", self._last_error)
                raise DialogueProviderError(self._last_error) from error
        finally:
            await _close_stream(request_stream)
            now = loop.time()
            # Only its owner may clear active state.  Old cleanup must never disturb a
            # newer request which has already replaced this generation.
            if generation == self._active_generation:
                self._active_stream = None
                self._active_generation = None
                self._active_started_at = None
                self._last_total_duration_ms = round((now - started_at) * 1_000, 1)
                if completed:
                    self._requests_completed += 1

    def invalidate_active(self) -> DialogueCancellationHandle | None:
        """Detach the current request synchronously and return its cleanup handle."""
        generation = self._active_generation
        if generation is None:
            return None
        request_stream = self._active_stream
        # This synchronous invalidation blocks late chunks even while close awaits I/O.
        self._active_generation = None
        self._active_stream = None
        self._active_started_at = None
        self._requests_cancelled += 1
        return _OpenAIDialogueCancellation(self, generation, request_stream)

    async def cancel(self) -> None:
        """Explicit cancellation intentionally targets whichever request is active now."""
        cleanup = self.invalidate_active()
        if cleanup is not None:
            await cleanup.close()

    def diagnostics(self) -> dict[str, object]:
        return {
            "provider": "openai-compatible",
            "base_url": self._base_url,
            "model": self._config.model or "not configured",
            "request_active": self._active_generation is not None,
            "request_generation": self._generation,
            "requests_started": self._requests_started,
            "requests_completed": self._requests_completed,
            "requests_cancelled": self._requests_cancelled,
            "requests_failed": self._requests_failed,
            "chunks_received": self._chunks_received,
            "characters_received": self._characters_received,
            "last_error": self._last_error,
            "last_finish_reason": self._last_finish_reason,
            "last_ttft_ms": self._last_ttft_ms,
            "last_total_duration_ms": self._last_total_duration_ms,
            "last_cancellation_latency_ms": self._last_cancellation_latency_ms,
            "live_text": self._live_text,
        }


async def _close_stream(stream: Any | None) -> None:
    if stream is None:
        return
    close = getattr(stream, "close", None) or getattr(stream, "aclose", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as error:  # Closing is best effort after invalidation.
        logger.debug("Dialogue stream close failed: %s", error)


def _chunk_content_and_finish_reason(chunk: Any) -> tuple[str, str]:
    """Tolerate OpenAI SDK objects and the deliberately malformed fake chunks in tests."""
    choices = _field(chunk, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return "", ""
    choice = choices[0]
    finish_reason = _field(choice, "finish_reason")
    delta = _field(choice, "delta")
    content = _field(delta, "content")
    return (content if isinstance(content, str) else "", finish_reason if isinstance(finish_reason, str) else "")


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
