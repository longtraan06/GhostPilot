"""Persistent client adapter for the self-hosted Nemotron streaming STT service."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
import json
import logging
from typing import Any, Literal, Protocol
from urllib.request import urlopen

from ..config import NemotronSTTConfig
from ..providers import STTEvent, STTProviderEvent, STTServiceEvent


logger = logging.getLogger(__name__)


class WebSocketLike(Protocol):
    async def send(self, message: bytes | str) -> None: ...
    async def recv(self) -> bytes | str: ...
    async def close(self) -> None: ...


ConnectFactory = Callable[[str], Awaitable[WebSocketLike]]
HealthFetcher = Callable[[str], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    """One ordered WebSocket write with its System 1 ownership metadata."""

    kind: Literal["audio", "control"]
    payload: bytes | str
    turn_id: str | None
    segment_id: int | None


async def _default_connect(url: str) -> WebSocketLike:
    try:
        from websockets.asyncio.client import connect
    except ImportError as error:
        raise RuntimeError("Install GhostPilot STT support: pip install -e '.[stt]'") from error
    return await connect(url, compression=None, max_size=1_048_576)


async def _default_health_fetch(url: str) -> dict[str, Any]:
    def fetch() -> dict[str, Any]:
        with urlopen(url, timeout=3.0) as response:  # noqa: S310 - configured local service URL
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("health response must be a JSON object")
        return payload

    return await asyncio.to_thread(fetch)


class NemotronSTTProvider:
    """Streams canonical PCM through one reconnecting, persistent WebSocket."""

    def __init__(
        self,
        config: NemotronSTTConfig | None = None,
        *,
        connect_factory: ConnectFactory | None = None,
        health_fetcher: HealthFetcher | None = None,
    ) -> None:
        self.config = config or NemotronSTTConfig()
        self._connect_factory = connect_factory or _default_connect
        self._health_fetcher = health_fetcher or _default_health_fetch
        self._send_queue: deque[OutboundMessage] = deque()
        self._send_condition = asyncio.Condition()
        # Controls are rare and get reserved bounded headroom. Audio remains
        # limited by send_queue_size and is the only traffic dropped on pressure.
        self._send_queue_capacity = self.config.send_queue_size + 8
        self._queued_audio_frames = 0
        self._events: asyncio.Queue[STTProviderEvent | None] = asyncio.Queue(128)
        self._socket: WebSocketLike | None = None
        self._supervisor_task: asyncio.Task[None] | None = None
        self._ready_event = asyncio.Event()
        self._closed = True
        self._active_turn_id: str | None = None
        self._active_segment_id: int | None = None
        self._connection_generation = 0
        self._connection_attempts = 0
        self._reconnect_count = 0
        self._connected = False
        self._ready = False
        self._health_status = "unknown"
        self._health: dict[str, Any] = {}
        self._last_error = ""
        self._frames_sent = 0
        self._audio_bytes_sent = 0
        self._dropped_frames = 0
        self._last_response_at: float | None = None
        self._last_audio_sent_at: float | None = None
        self._last_response_type = "none"
        self._last_control_sent = "none"
        self._messages_received = 0
        self._partial_responses = 0
        self._final_responses = 0
        self._ignored_responses = 0
        self._last_ignored_reason = ""
        self._segment_frames_queued = 0
        self._segment_bytes_queued = 0
        self._segment_frames_sent = 0
        self._segment_bytes_sent = 0
        self._segment_partial_responses = 0
        self._segment_final_responses = 0
        self._trace: deque[str] = deque(maxlen=24)

    async def connect(self) -> None:
        if self._supervisor_task and not self._supervisor_task.done():
            return
        self._closed = False
        self._ready_event.clear()
        self._supervisor_task = asyncio.create_task(self._connection_supervisor())
        # Temporary service loss must not prevent the rest of System 1 from
        # starting. The supervisor continues bounded reconnect attempts.
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                self._ready_event.wait(), timeout=self.config.ready_timeout_seconds
            )

    async def send_audio(self, audio: bytes) -> None:
        if not audio:
            return
        async with self._send_condition:
            if (
                self._queued_audio_frames >= self.config.send_queue_size
                or len(self._send_queue) >= self._send_queue_capacity
            ):
                self._record_audio_drop("send queue full")
                return
            self._send_queue.append(
                OutboundMessage(
                    "audio", audio, self._active_turn_id, self._active_segment_id
                )
            )
            self._queued_audio_frames += 1
            self._segment_frames_queued += 1
            self._segment_bytes_queued += len(audio)
            self._send_condition.notify()

    async def start_segment(self, turn_id: str, segment_id: int) -> None:
        self._active_turn_id = turn_id
        self._active_segment_id = segment_id
        self._segment_frames_queued = 0
        self._segment_bytes_queued = 0
        self._segment_frames_sent = 0
        self._segment_bytes_sent = 0
        self._segment_partial_responses = 0
        self._segment_final_responses = 0
        self._trace_event(f"segment started: {turn_id} / {segment_id}")
        logger.info("STT segment started", extra={"turn_id": turn_id, "segment_id": segment_id})

    async def end_segment(self) -> None:
        await self._enqueue_control("segment_end")
        self._trace_event(
            f"segment_end queued: {self._segment_frames_queued} frames / "
            f"{self._segment_bytes_queued} bytes"
        )
        logger.info("STT segment_end queued", extra={"segment_id": self._active_segment_id})

    async def commit_turn(self) -> None:
        await self._enqueue_control("commit")
        logger.info("STT turn commit queued", extra={"turn_id": self._active_turn_id})
        self._active_turn_id = None
        self._active_segment_id = None

    async def reset(self) -> None:
        self._active_turn_id = None
        self._active_segment_id = None
        await self._discard_send_queue("explicit reset")
        await self._enqueue_control("reset")
        await self._emit_service("reset", "STT session reset requested")
        logger.info("STT reset queued")

    async def ping(self) -> None:
        await self._enqueue_control("ping")

    async def reconnect(self) -> None:
        socket = self._socket
        if socket is not None:
            await socket.close()

    async def events(self) -> AsyncIterator[STTProviderEvent]:
        while (event := await self._events.get()) is not None:
            yield event

    def diagnostics(self) -> dict[str, object]:
        now = asyncio.get_running_loop().time()
        response_age_ms = (
            round((now - self._last_response_at) * 1_000, 1)
            if self._last_response_at is not None
            else None
        )
        audio_send_age_ms = (
            round((now - self._last_audio_sent_at) * 1_000, 1)
            if self._last_audio_sent_at is not None
            else None
        )
        return {
            "provider": "nemotron",
            "connected": self._connected,
            "ready": self._ready,
            "health_status": self._health_status,
            "model": self._health.get("model", ""),
            "device": self._health.get("device", ""),
            "lookahead": self._health.get("lookahead"),
            "streaming_latency_ms": self._health.get("streaming_latency_ms"),
            "warmed_up": self._health.get("warmed_up", False),
            "reconnect_count": self._reconnect_count,
            "connection_generation": self._connection_generation,
            "last_error": self._last_error,
            "frames_sent": self._frames_sent,
            "audio_bytes_sent": self._audio_bytes_sent,
            "send_queue_depth": len(self._send_queue),
            "send_queue_capacity": self._send_queue_capacity,
            "audio_queue_capacity": self.config.send_queue_size,
            "stt_dropped_frames": self._dropped_frames,
            "latest_response_age_ms": response_age_ms,
            "latest_audio_send_age_ms": audio_send_age_ms,
            "active_turn_id": self._active_turn_id,
            "active_segment_id": self._active_segment_id,
            "last_response_type": self._last_response_type,
            "last_control_sent": self._last_control_sent,
            "messages_received": self._messages_received,
            "partial_responses": self._partial_responses,
            "final_responses": self._final_responses,
            "ignored_responses": self._ignored_responses,
            "last_ignored_reason": self._last_ignored_reason,
            "segment_frames_queued": self._segment_frames_queued,
            "segment_bytes_queued": self._segment_bytes_queued,
            "segment_frames_sent": self._segment_frames_sent,
            "segment_bytes_sent": self._segment_bytes_sent,
            "segment_partial_responses": self._segment_partial_responses,
            "segment_final_responses": self._segment_final_responses,
            "trace": list(self._trace),
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        socket, self._socket = self._socket, None
        if socket is not None:
            with suppress(Exception):
                await socket.close()
        task, self._supervisor_task = self._supervisor_task, None
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._connected = False
        self._ready = False
        self._ready_event.clear()
        async with self._send_condition:
            self._send_condition.notify_all()
        if self._events.full():
            with suppress(asyncio.QueueEmpty):
                self._events.get_nowait()
        self._events.put_nowait(None)

    async def _connection_supervisor(self) -> None:
        delay = self.config.reconnect_initial_seconds
        while not self._closed:
            self._connection_attempts += 1
            if self._connection_attempts > 1:
                self._reconnect_count += 1
            await self._emit_service("connecting", f"attempt {self._connection_attempts}")
            try:
                await self._validate_health()
                socket = await asyncio.wait_for(
                    self._connect_factory(self.config.ws_url),
                    timeout=self.config.connect_timeout_seconds,
                )
                self._connection_generation += 1
                generation = self._connection_generation
                self._socket = socket
                self._connected = True
                self._ready = False
                self._ready_event.clear()
                self._last_error = ""
                logger.info("STT connection opened", extra={"generation": generation})
                self._trace_event(f"connection {generation} opened")
                await self._emit_service("connected", connection_generation=generation)

                # A fresh/reconnected socket must not inherit decoder state.
                await socket.send(json.dumps({"type": "reset"}))
                self._last_control_sent = "reset"
                self._trace_event("control sent: reset (connection initialization)")
                logger.info(
                    "STT control sent: reset (connection initialization, generation=%s)",
                    generation,
                )
                sender = asyncio.create_task(self._send_loop(socket, generation))
                receiver = asyncio.create_task(self._receive_loop(socket, generation))
                try:
                    done, _ = await asyncio.wait(
                        {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        error = task.exception()
                        if error:
                            raise error
                finally:
                    for task in (sender, receiver):
                        if not task.done():
                            task.cancel()
                    for task in (sender, receiver):
                        with suppress(asyncio.CancelledError, Exception):
                            await task
                raise ConnectionError("STT WebSocket closed")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._last_error = str(error)
                self._trace_event(f"connection error: {error}")
                logger.warning("STT connection unavailable: %s", error)
                await self._emit_service("error", str(error))
            finally:
                self._connected = False
                self._ready = False
                self._ready_event.clear()
                self._socket = None
                self._active_turn_id = None
                self._active_segment_id = None
                await self._discard_send_queue("connection changed")
                await self._emit_service("disconnected", self._last_error)
                logger.info("STT connection closed")
            if not self._closed:
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.reconnect_max_seconds)

    async def _send_loop(self, socket: WebSocketLike, generation: int) -> None:
        while generation == self._connection_generation and not self._closed:
            async with self._send_condition:
                while not self._send_queue and not self._closed:
                    await self._send_condition.wait()
                if self._closed:
                    return
                outbound = self._send_queue.popleft()
                if outbound.kind == "audio":
                    self._queued_audio_frames -= 1
                self._send_condition.notify_all()
            await socket.send(outbound.payload)
            if outbound.kind == "audio":
                message = outbound.payload
                assert isinstance(message, bytes)
                self._frames_sent += 1
                self._audio_bytes_sent += len(message)
                if (
                    outbound.turn_id == self._active_turn_id
                    and outbound.segment_id == self._active_segment_id
                ):
                    self._segment_frames_sent += 1
                    self._segment_bytes_sent += len(message)
                self._last_audio_sent_at = asyncio.get_running_loop().time()
                if self._segment_frames_sent == 1:
                    self._trace_event(
                        f"first PCM frame sent: {len(message)} bytes / connection {generation}"
                    )
                    logger.info(
                        "STT first PCM frame sent",
                        extra={
                            "turn_id": self._active_turn_id,
                            "segment_id": outbound.segment_id,
                            "bytes": len(message),
                        },
                    )
                elif self._segment_frames_sent % 50 == 0:
                    logger.debug(
                        "STT audio sent: segment=%s frames=%s bytes=%s queue=%s",
                        self._active_segment_id,
                        self._segment_frames_sent,
                        self._segment_bytes_sent,
                        len(self._send_queue),
                    )
            else:
                message = outbound.payload
                assert isinstance(message, str)
                try:
                    control_type = str(json.loads(message).get("type", "unknown"))
                except (json.JSONDecodeError, AttributeError):
                    control_type = "malformed-control"
                self._last_control_sent = control_type
                self._trace_event(
                    f"control sent: {control_type} after "
                    f"{self._segment_frames_sent} frames / {self._segment_bytes_sent} bytes"
                )
                logger.info(
                    "STT control sent: %s (segment=%s frames=%s bytes=%s)",
                    control_type,
                    outbound.segment_id,
                    self._segment_frames_sent,
                    self._segment_bytes_sent,
                )

    async def _receive_loop(self, socket: WebSocketLike, generation: int) -> None:
        while generation == self._connection_generation and not self._closed:
            message = await socket.recv()
            await self._handle_message(message, generation)

    async def _handle_message(self, message: bytes | str, generation: int) -> None:
        if generation != self._connection_generation:
            self._ignore_response(
                f"stale connection generation {generation}; active {self._connection_generation}"
            )
            return
        self._last_response_at = asyncio.get_running_loop().time()
        self._messages_received += 1
        try:
            raw = message.decode("utf-8") if isinstance(message, bytes) else message
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
                raise ValueError("STT message must be an object with a string type")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self._last_response_type = "malformed"
            self._trace_event(f"malformed response: {error}")
            await self._provider_error(f"malformed STT response: {error}", generation)
            return

        message_type = payload["type"]
        self._last_response_type = message_type
        logger.debug("STT response received: type=%s payload=%s", message_type, payload)
        if message_type == "ready":
            self._ready = True
            self._ready_event.set()
            self._trace_event(f"response received: ready / connection {generation}")
            await self._emit_service("ready", connection_generation=generation)
            logger.info("STT ready", extra={"generation": generation})
            return
        if message_type in {"partial", "final"}:
            text = payload.get("text")
            segment_id = payload.get("segment_id")
            if not isinstance(text, str) or not isinstance(segment_id, int):
                self._ignore_response(f"invalid {message_type}: missing text or integer segment_id")
                await self._provider_error(
                    f"invalid {message_type} response", generation
                )
                return
            if self._active_turn_id is None or self._active_segment_id is None:
                self._ignore_response(f"{message_type} ignored: no active turn/segment")
                return
            if segment_id != self._active_segment_id:
                self._ignore_response(
                    f"{message_type} segment {segment_id} != active {self._active_segment_id}"
                )
                return
            if message_type == "partial":
                self._partial_responses += 1
                self._segment_partial_responses += 1
            else:
                self._final_responses += 1
                self._segment_final_responses += 1
            self._trace_event(
                f"response received: {message_type} / segment {segment_id} / {len(text)} chars"
            )
            await self._offer_event(
                STTEvent(
                    text=text,
                    is_final=message_type == "final",
                    turn_id=self._active_turn_id,
                    segment_id=segment_id,
                )
            )
            logger.debug("STT %s received: %s", message_type, text)
            return
        if message_type == "error":
            code = payload.get("code", "unknown")
            detail = f"{code}: {payload.get('message', 'STT service error')}"
            await self._provider_error(detail, generation)
            return
        # turn_final is an acknowledgement/snapshot only. EndpointDetector is
        # the sole authority allowed to commit a GhostPilot user turn.
        if message_type in {
            "turn_final",
            "pong",
            "reset",
            "reset_ack",
            "reset_ok",
            "commit_ok",
            "segment_end_ok",
            "ack",
        }:
            self._trace_event(f"response received: {message_type}")
            return
        self._ignore_response(f"unknown response type: {message_type}")

    async def _validate_health(self) -> None:
        health = await self._health_fetcher(self.config.health_url)
        self._health = health
        mismatches = []
        if health.get("ok") is not True:
            mismatches.append("ok must be true")
        if health.get("sample_rate") != 16_000:
            mismatches.append("sample_rate must be 16000")
        if health.get("lookahead") != 1:
            mismatches.append("lookahead must be 1")
        if health.get("warmed_up") is not True:
            mismatches.append("warmed_up must be true")
        if mismatches:
            self._health_status = "unhealthy"
            raise RuntimeError("invalid STT health: " + ", ".join(mismatches))
        self._health_status = "healthy"

    async def _enqueue_control(self, message_type: str) -> None:
        message = json.dumps({"type": message_type}, separators=(",", ":"))
        async with self._send_condition:
            while len(self._send_queue) >= self._send_queue_capacity:
                audio_index = next(
                    (
                        index
                        for index, queued in enumerate(self._send_queue)
                        if queued.kind == "audio"
                    ),
                    None,
                )
                if audio_index is not None:
                    del self._send_queue[audio_index]
                    self._queued_audio_frames -= 1
                    self._record_audio_drop(f"evicted for {message_type} control")
                    break
                if self._closed:
                    return
                await self._send_condition.wait()
            self._send_queue.append(
                OutboundMessage(
                    "control", message, self._active_turn_id, self._active_segment_id
                )
            )
            self._trace_event(f"control queued: {message_type}")
            logger.info(
                "STT control queued: %s (turn=%s segment=%s)",
                message_type,
                self._active_turn_id,
                self._active_segment_id,
            )
            self._send_condition.notify()

    async def _discard_send_queue(self, reason: str) -> None:
        async with self._send_condition:
            while self._send_queue:
                message = self._send_queue.popleft()
                if message.kind == "audio":
                    self._queued_audio_frames -= 1
                    self._record_audio_drop(f"discarded: {reason}")
            self._send_condition.notify_all()

    def _record_audio_drop(self, reason: str) -> None:
        self._dropped_frames += 1
        self._trace_event(f"audio dropped: {reason}")
        if self._dropped_frames == 1 or self._dropped_frames % 50 == 0:
            logger.warning(
                "STT audio frame dropped: %s (total=%s)", reason, self._dropped_frames
            )

    async def _provider_error(self, detail: str, generation: int) -> None:
        self._last_error = detail
        logger.warning("STT provider error: %s", detail)
        await self._emit_service("error", detail, generation)

    def _ignore_response(self, reason: str) -> None:
        self._ignored_responses += 1
        self._last_ignored_reason = reason
        self._trace_event(f"response ignored: {reason}")
        logger.warning("STT response ignored: %s", reason)

    def _trace_event(self, message: str) -> None:
        self._trace.append(message)

    async def _emit_service(
        self,
        status: str,
        detail: str = "",
        connection_generation: int | None = None,
    ) -> None:
        await self._offer_event(
            STTServiceEvent(
                status=status,  # type: ignore[arg-type]
                detail=detail,
                connection_generation=(
                    self._connection_generation
                    if connection_generation is None
                    else connection_generation
                ),
            )
        )

    async def _offer_event(self, event: STTProviderEvent) -> None:
        if self._events.full():
            with suppress(asyncio.QueueEmpty):
                self._events.get_nowait()
        self._events.put_nowait(event)
