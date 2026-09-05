"""
GhostPilot Nemotron Streaming STT Service

Backend service for:
    nvidia/nemotron-speech-streaming-en-0.6b

System 1 contract:
- FastAPI + WebSocket
- CUDA device: cuda:0
- Lookahead: 1 (~160 ms streaming latency mode)
- Input audio: raw PCM16 little-endian, mono, 16 kHz
- Output: snapshot-style partial/final transcript events
- STT only: no VAD, no endpoint detection, no dialogue logic

Run:
    pip install fastapi "uvicorn[standard]" numpy torch transformers
    python nemotron_stt_service.py

WebSocket:
    ws://HOST:8001/v1/stt/stream

Binary messages:
    Raw PCM16 mono 16 kHz bytes.

JSON control messages:
    {"type": "segment_end"}
        Finalize the current VAD speech segment.
        The service emits a "final" transcript snapshot, but does NOT decide
        whether the GhostPilot user turn is complete.

    {"type": "commit"}
        Finalize the current segment, emit "turn_final", then reset transcript
        state for the next GhostPilot turn.

    {"type": "reset"}
        Discard current decoder/transcript state and start clean.

    {"type": "ping"}
        Returns {"type": "pong"}.

Important:
- VAD and EndpointDetector remain responsibilities of GhostPilot System 1.
- `segment_end` is intended to be called by the future STTProvider adapter when
  System 1 observes a VAD speech-stop boundary and wants a stable segment result.
- A resumed speech segment can continue the same GhostPilot turn; this service
  preserves an accumulated turn transcript across segments until `commit/reset`.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import traceback
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
DEVICE = "cuda:0"
LOOKAHEAD = 1
HOST = "0.0.0.0"
PORT = 8001

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
MAX_BINARY_MESSAGE_BYTES = SAMPLE_RATE * SAMPLE_WIDTH_BYTES * 2  # <= 2 s/message

# GhostPilot is a personal agent. Start conservatively with one active GPU
# decoder session; this can be relaxed after concurrency benchmarking.
MAX_ACTIVE_SESSIONS = 1


def _pcm16le_to_float32(data: bytes) -> np.ndarray:
    if len(data) % SAMPLE_WIDTH_BYTES:
        raise ValueError("PCM16 payload length must be divisible by 2")
    if not data:
        return np.empty(0, dtype=np.float32)
    pcm = np.frombuffer(data, dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def _join_snapshot(prefix: str, segment: str) -> str:
    prefix = prefix.strip()
    segment = segment.strip()
    if not prefix:
        return segment
    if not segment:
        return prefix

    # RNNT text pieces normally contain their own spacing. This fallback keeps
    # full-turn snapshots readable when a segment starts without a leading space.
    if segment[0] in ",.!?;:":
        return prefix + segment
    return prefix + " " + segment


@dataclass(slots=True)
class ModelRuntime:
    processor: Any
    model: Any
    dtype: torch.dtype
    warmed_up: bool = False
    warmup_seconds: float | None = None


MODEL_RUNTIME: ModelRuntime | None = None
SESSION_SEMAPHORE = asyncio.Semaphore(MAX_ACTIVE_SESSIONS)


class StreamingSegmentBuilder:
    """
    Incrementally reproduces the model's cache-aware streaming audio windows.

    The benchmark builds the first segment from:
        processor.num_samples_first_audio_chunk

    Later windows follow the processor's mel-frame/hop geometry and can include
    overlap required by the feature extractor. This builder performs the same
    indexing incrementally for live PCM.
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        fe = processor.feature_extractor

        if fe.sampling_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"Expected model sample rate {SAMPLE_RATE}, got {fe.sampling_rate}"
            )

        self.first_n = int(processor.num_samples_first_audio_chunk)
        self.chunk_n = int(processor.num_samples_per_audio_chunk)
        self.first_mel = int(processor.num_mel_frames_first_audio_chunk)
        self.chunk_mel = int(processor.num_mel_frames_per_audio_chunk)
        self.hop = int(fe.hop_length)
        self.n_fft = int(fe.n_fft)

        self._audio = np.empty(0, dtype=np.float32)
        self._base_index = 0
        self._first_emitted = False
        self._next_start = 0
        self._mel_idx = self.first_mel
        self._has_any_audio = False

    @property
    def has_audio(self) -> bool:
        return self._has_any_audio

    def reset(self) -> None:
        self._audio = np.empty(0, dtype=np.float32)
        self._base_index = 0
        self._first_emitted = False
        self._next_start = 0
        self._mel_idx = self.first_mel
        self._has_any_audio = False

    def append(self, samples: np.ndarray) -> list[tuple[np.ndarray, bool]]:
        if samples.size:
            self._has_any_audio = True
            self._audio = np.concatenate((self._audio, samples))
        return self._drain_ready()

    def flush(self) -> list[tuple[np.ndarray, bool]]:
        """
        Emit a final zero-padded window when needed so short utterances and
        trailing speech are not dropped.
        """
        out = self._drain_ready()

        if not self._has_any_audio:
            return out

        absolute_end = self._base_index + len(self._audio)

        if not self._first_emitted:
            segment = self._slice_absolute(0, self.first_n, pad=True)
            self._first_emitted = True
            self._next_start = self._mel_idx * self.hop - self.n_fft // 2
            self._mel_idx += self.chunk_mel
            out.append((segment, True))
            self._trim()
            return out

        if absolute_end > self._next_start:
            segment = self._slice_absolute(
                self._next_start, self._next_start + self.chunk_n, pad=True
            )
            self._next_start = self._mel_idx * self.hop - self.n_fft // 2
            self._mel_idx += self.chunk_mel
            out.append((segment, False))
            self._trim()

        return out

    def _drain_ready(self) -> list[tuple[np.ndarray, bool]]:
        out: list[tuple[np.ndarray, bool]] = []

        while True:
            absolute_end = self._base_index + len(self._audio)

            if not self._first_emitted:
                if absolute_end < self.first_n:
                    break
                segment = self._slice_absolute(0, self.first_n, pad=False)
                self._first_emitted = True
                self._next_start = self._mel_idx * self.hop - self.n_fft // 2
                self._mel_idx += self.chunk_mel
                out.append((segment, True))
                self._trim()
                continue

            required_end = self._next_start + self.chunk_n
            if absolute_end < required_end:
                break

            segment = self._slice_absolute(self._next_start, required_end, pad=False)
            self._next_start = self._mel_idx * self.hop - self.n_fft // 2
            self._mel_idx += self.chunk_mel
            out.append((segment, False))
            self._trim()

        return out

    def _slice_absolute(self, start: int, end: int, *, pad: bool) -> np.ndarray:
        local_start = start - self._base_index
        local_end = end - self._base_index

        left_pad = max(0, -local_start)
        right_pad = max(0, local_end - len(self._audio))

        a = max(0, local_start)
        b = min(len(self._audio), local_end)
        segment = self._audio[a:b]

        if pad and (left_pad or right_pad):
            segment = np.pad(segment, (left_pad, right_pad))

        expected = end - start
        if len(segment) != expected:
            raise RuntimeError(
                f"stream segment has {len(segment)} samples, expected {expected}"
            )

        return np.ascontiguousarray(segment, dtype=np.float32)

    def _trim(self) -> None:
        # Keep only audio that can still be referenced by the next overlapping
        # model window.
        keep_from = max(0, self._next_start)
        drop = keep_from - self._base_index
        if drop <= 0:
            return
        drop = min(drop, len(self._audio))
        self._audio = self._audio[drop:]
        self._base_index += drop


class DecoderWorker:
    """
    One cache-aware RNNT generation call for one VAD speech segment.

    A segment is finalized by placing None into the feature queue. Text pieces
    are forwarded to the asyncio event loop as they are emitted.
    """

    def __init__(
        self,
        runtime: ModelRuntime,
        loop: asyncio.AbstractEventLoop,
        on_event: Callable[[dict[str, Any]], None],
        segment_id: int,
    ) -> None:
        self.runtime = runtime
        self.loop = loop
        self.on_event = on_event
        self.segment_id = segment_id

        self._features: queue.Queue[tuple[np.ndarray, bool] | None] = queue.Queue(maxsize=64)
        self._thread = threading.Thread(
            target=self._run,
            name=f"nemotron-segment-{segment_id}",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def put(self, segment: np.ndarray, is_first: bool) -> None:
        self.start()
        self._features.put((segment, is_first))

    def finish(self) -> None:
        if not self._started:
            return
        self._features.put(None)

    def _emit(self, event: dict[str, Any]) -> None:
        self.loop.call_soon_threadsafe(self.on_event, event)

    def _run(self) -> None:
        try:
            first_item = self._features.get()
            if first_item is None:
                self._emit({"kind": "done", "segment_id": self.segment_id})
                return

            first_audio, is_first = first_item
            if not is_first:
                raise RuntimeError("decoder must start with the first streaming audio window")

            processor = self.runtime.processor
            model = self.runtime.model
            dtype = self.runtime.dtype

            first_inputs = processor(
                first_audio,
                sampling_rate=SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=True,
                return_tensors="pt",
            ).to(DEVICE, dtype=dtype)

            first_feats = first_inputs["input_features"][
                :, : processor.num_mel_frames_first_audio_chunk, :
            ]

            def feature_generator():
                yield first_feats
                while True:
                    item = self._features.get()
                    if item is None:
                        return

                    audio, item_is_first = item
                    if item_is_first:
                        raise RuntimeError("received a second first-chunk marker")

                    x = processor(
                        audio,
                        sampling_rate=SAMPLE_RATE,
                        is_streaming=True,
                        is_first_audio_chunk=False,
                        return_tensors="pt",
                    )
                    yield x["input_features"].to(DEVICE, dtype=dtype)

            streamer = TextIteratorStreamer(
                processor.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True,
            )

            generation_error: list[str] = []

            def generate() -> None:
                try:
                    with torch.inference_mode():
                        model.generate(
                            **{
                                **first_inputs,
                                "input_features": feature_generator(),
                                "streamer": streamer,
                            }
                        )
                except Exception:
                    generation_error.append(traceback.format_exc())
                    streamer.end()

            generate_thread = threading.Thread(
                target=generate,
                name=f"nemotron-generate-{self.segment_id}",
                daemon=True,
            )
            generate_thread.start()

            for piece in streamer:
                if piece:
                    self._emit(
                        {
                            "kind": "piece",
                            "segment_id": self.segment_id,
                            "text": piece,
                        }
                    )

            generate_thread.join()

            if generation_error:
                raise RuntimeError(generation_error[0])

            self._emit({"kind": "done", "segment_id": self.segment_id})

        except Exception:
            self._emit(
                {
                    "kind": "error",
                    "segment_id": self.segment_id,
                    "error": traceback.format_exc(),
                }
            )


class STTSession:
    """
    WebSocket-facing STT state.

    A GhostPilot turn may contain multiple VAD speech segments. Each segment is
    decoded independently for clean finality, while this class keeps a
    full-turn transcript prefix so every outgoing transcript remains a snapshot.
    """

    def __init__(self, websocket: WebSocket, runtime: ModelRuntime) -> None:
        self.websocket = websocket
        self.runtime = runtime
        self.loop = asyncio.get_running_loop()

        self.builder = StreamingSegmentBuilder(runtime.processor)
        self.outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)

        self.turn_text = ""
        self.segment_text = ""
        self.segment_id = 0
        self.decoder: DecoderWorker | None = None
        self.segment_done = asyncio.Event()
        self.closed = False

    async def send_ready(self) -> None:
        await self.websocket.send_json(
            {
                "type": "ready",
                "model": MODEL_ID,
                "device": DEVICE,
                "lookahead": LOOKAHEAD,
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "sample_format": "pcm_s16le",
                "streaming_latency_ms": int(self.runtime.processor.streaming_latency_ms),
                "warmed_up": self.runtime.warmed_up,
            }
        )

    def _ensure_decoder(self) -> DecoderWorker:
        if self.decoder is None:
            self.segment_id += 1
            self.segment_text = ""
            self.segment_done = asyncio.Event()
            self.decoder = DecoderWorker(
                self.runtime,
                self.loop,
                self._on_decoder_event,
                self.segment_id,
            )
        return self.decoder

    def _on_decoder_event(self, event: dict[str, Any]) -> None:
        if self.closed:
            return

        if event.get("segment_id") != self.segment_id:
            # Ignore a late event from a decoder that no longer owns the active
            # segment.
            return

        kind = event["kind"]

        if kind == "piece":
            self.segment_text += event["text"]
            snapshot = _join_snapshot(self.turn_text, self.segment_text)
            self._put_outgoing(
                {
                    "type": "partial",
                    "text": snapshot,
                    "segment_id": self.segment_id,
                }
            )
            return

        if kind == "done":
            snapshot = _join_snapshot(self.turn_text, self.segment_text)
            self.turn_text = snapshot
            self._put_outgoing(
                {
                    "type": "final",
                    "text": snapshot,
                    "segment_id": self.segment_id,
                }
            )
            self.decoder = None
            self.builder.reset()
            self.segment_done.set()
            return

        if kind == "error":
            self._put_outgoing(
                {
                    "type": "error",
                    "code": "decoder_error",
                    "message": event["error"],
                    "segment_id": self.segment_id,
                }
            )
            self.decoder = None
            self.builder.reset()
            self.segment_done.set()

    def _put_outgoing(self, event: dict[str, Any]) -> None:
        if self.outgoing.full():
            # Text events are tiny, but never allow unbounded memory growth.
            try:
                self.outgoing.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self.outgoing.put_nowait(event)

    async def sender_loop(self) -> None:
        while not self.closed:
            event = await self.outgoing.get()
            await self.websocket.send_json(event)

    async def feed_binary(self, data: bytes) -> None:
        if len(data) > MAX_BINARY_MESSAGE_BYTES:
            raise ValueError(
                f"audio message too large ({len(data)} bytes); "
                f"max is {MAX_BINARY_MESSAGE_BYTES}"
            )

        samples = _pcm16le_to_float32(data)
        for segment, is_first in self.builder.append(samples):
            self._ensure_decoder().put(segment, is_first)

    async def finalize_segment(self) -> None:
        if not self.builder.has_audio and self.decoder is None:
            # Preserve protocol determinism: an empty segment simply returns the
            # current turn snapshot as final.
            await self.outgoing.put(
                {
                    "type": "final",
                    "text": self.turn_text,
                    "segment_id": self.segment_id,
                }
            )
            return

        for segment, is_first in self.builder.flush():
            self._ensure_decoder().put(segment, is_first)

        decoder = self.decoder
        if decoder is None:
            return

        decoder.finish()
        await self.segment_done.wait()

    async def commit_turn(self) -> None:
        await self.finalize_segment()
        await self.outgoing.put(
            {
                "type": "turn_final",
                "text": self.turn_text,
            }
        )
        self.turn_text = ""
        self.segment_text = ""
        self.builder.reset()

    async def reset(self) -> None:
        if self.decoder is not None:
            # End the model iterator cleanly, but discard the resulting
            # transcript by invalidating the active segment id.
            old = self.decoder
            self.segment_id += 1
            self.decoder = None
            old.finish()

        self.turn_text = ""
        self.segment_text = ""
        self.builder.reset()
        self.segment_done.set()
        await self.outgoing.put({"type": "reset_ok"})

    async def close(self) -> None:
        self.closed = True
        if self.decoder is not None:
            self.decoder.finish()
            self.decoder = None



def _warmup_streaming_model(runtime: ModelRuntime) -> float:
    """
    Execute one real cache-aware streaming generate() pass on CUDA.

    This is intentionally NOT a benchmark. Its only purpose is to initialize
    CUDA kernels, allocator paths, processor state, RNNT generation internals,
    and TextIteratorStreamer before the service starts accepting traffic.

    The warm-up uses synthetic silence and the exact configured lookahead.
    """
    processor = runtime.processor
    model = runtime.model
    dtype = runtime.dtype

    processor.set_num_lookahead_tokens(LOOKAHEAD)

    # Use enough synthetic audio to exercise the first streaming window and
    # several subsequent windows. Silence is sufficient for kernel warm-up.
    warmup_seconds = 1.5
    audio = np.zeros(int(SAMPLE_RATE * warmup_seconds), dtype=np.float32)

    first_n = int(processor.num_samples_first_audio_chunk)
    chunk_n = int(processor.num_samples_per_audio_chunk)
    first_mel = int(processor.num_mel_frames_first_audio_chunk)
    chunk_mel = int(processor.num_mel_frames_per_audio_chunk)
    hop = int(processor.feature_extractor.hop_length)
    n_fft = int(processor.feature_extractor.n_fft)

    first_audio = audio[:first_n]
    if len(first_audio) < first_n:
        first_audio = np.pad(first_audio, (0, first_n - len(first_audio)))

    first_inputs = processor(
        first_audio,
        sampling_rate=SAMPLE_RATE,
        is_streaming=True,
        is_first_audio_chunk=True,
        return_tensors="pt",
    ).to(DEVICE, dtype=dtype)

    def feature_generator():
        yield first_inputs["input_features"][
            :, : processor.num_mel_frames_first_audio_chunk, :
        ]

        mel_idx = first_mel
        start_idx = mel_idx * hop - n_fft // 2

        while start_idx < len(audio):
            end_idx = start_idx + chunk_n
            segment = audio[max(start_idx, 0):max(start_idx, 0) + chunk_n]

            if start_idx < 0:
                segment = np.pad(segment, (-start_idx, 0))

            if len(segment) < chunk_n:
                segment = np.pad(segment, (0, chunk_n - len(segment)))

            x = processor(
                np.ascontiguousarray(segment, dtype=np.float32),
                sampling_rate=SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )
            yield x["input_features"].to(DEVICE, dtype=dtype)

            mel_idx += chunk_mel
            start_idx = mel_idx * hop - n_fft // 2

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
    )

    error: list[str] = []

    def generate():
        try:
            with torch.inference_mode():
                model.generate(
                    **{
                        **first_inputs,
                        "input_features": feature_generator(),
                        "streamer": streamer,
                    }
                )
        except Exception:
            error.append(traceback.format_exc())
            streamer.end()

    started = time.perf_counter()
    thread = threading.Thread(
        target=generate,
        name="nemotron-warmup",
        daemon=True,
    )
    thread.start()

    # Drain the streamer so generation can proceed exactly as it will in a
    # real session. Output is deliberately discarded.
    for _ in streamer:
        pass

    thread.join()

    if error:
        raise RuntimeError(f"Nemotron warm-up failed:\n{error[0]}")

    torch.cuda.synchronize(DEVICE)
    return time.perf_counter() - started


async def _load_model() -> ModelRuntime:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    processor = await asyncio.to_thread(AutoProcessor.from_pretrained, MODEL_ID)
    processor.set_num_lookahead_tokens(LOOKAHEAD)

    model = await asyncio.to_thread(AutoModelForRNNT.from_pretrained, MODEL_ID)
    model = model.to(DEVICE)
    model.eval()

    dtype = model.dtype

    runtime = ModelRuntime(
        processor=processor,
        model=model,
        dtype=dtype,
    )

    # Do not accept traffic until the exact streaming inference path has run
    # once on CUDA.
    warmup_seconds = await asyncio.to_thread(_warmup_streaming_model, runtime)
    runtime.warmed_up = True
    runtime.warmup_seconds = warmup_seconds

    return runtime


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL_RUNTIME

    MODEL_RUNTIME = await _load_model()
    yield

    MODEL_RUNTIME = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="GhostPilot Nemotron Streaming STT",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    runtime = MODEL_RUNTIME
    return {
        "ok": runtime is not None,
        "model": MODEL_ID,
        "device": DEVICE,
        "lookahead": LOOKAHEAD,
        "sample_rate": SAMPLE_RATE,
        "streaming_latency_ms": (
            int(runtime.processor.streaming_latency_ms) if runtime else None
        ),
        "warmed_up": runtime.warmed_up if runtime else False,
        "warmup_seconds": runtime.warmup_seconds if runtime else None,
    }


@app.websocket("/v1/stt/stream")
async def stt_stream(websocket: WebSocket) -> None:
    runtime = MODEL_RUNTIME
    if runtime is None:
        await websocket.close(code=1013, reason="model is not ready")
        return

    if SESSION_SEMAPHORE.locked():
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "code": "busy",
                "message": "STT GPU session is busy",
            }
        )
        await websocket.close(code=1013)
        return

    await SESSION_SEMAPHORE.acquire()
    await websocket.accept()

    session = STTSession(websocket, runtime)
    sender_task = asyncio.create_task(session.sender_loop())

    try:
        await session.send_ready()

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            binary = message.get("bytes")
            text = message.get("text")

            if binary is not None:
                try:
                    await session.feed_binary(binary)
                except ValueError as exc:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "bad_audio",
                            "message": str(exc),
                        }
                    )
                continue

            if text is None:
                continue

            try:
                control = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "bad_json",
                        "message": "control messages must be valid JSON",
                    }
                )
                continue

            kind = control.get("type")

            if kind == "segment_end":
                await session.finalize_segment()
            elif kind == "commit":
                await session.commit_turn()
            elif kind == "reset":
                await session.reset()
            elif kind == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "unknown_control",
                        "message": f"unknown control type: {kind!r}",
                    }
                )

    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        SESSION_SEMAPHORE.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "nemotron_stt_service:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
