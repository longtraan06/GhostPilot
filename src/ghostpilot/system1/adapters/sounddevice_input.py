"""sounddevice adapter; all PortAudio-specific code is isolated here."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from ..audio import AudioFrame
from ..config import AudioConfig


class SoundDeviceAudioInput:
    """Captures fixed, canonical PCM16 frames via a bounded asyncio queue."""

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()
        self._queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue(self.config.queue_size)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: Any = None
        self._capture_sample_rate: int | None = None
        self._capture_channels: int | None = None
        self._closed = True
        self._sequence = 0
        self.frames_dropped = 0

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError as error:
            raise RuntimeError("Install GhostPilot audio support: pip install -e '.[audio]'") from error
        self._loop = asyncio.get_running_loop()
        self._closed = False
        device_info = sd.query_devices(self.config.device, kind="input")
        self._capture_sample_rate = round(device_info["default_samplerate"])
        self._capture_channels = min(2, device_info["max_input_channels"])
        if self._capture_channels < 1:
            raise RuntimeError("selected audio device has no input channels")
        capture_samples_per_frame = round(
            self._capture_sample_rate * self.config.frame_duration_ms / 1_000
        )

        def callback(indata: Any, frames: int, _time_info: Any, status: Any) -> None:
            if status:
                # PortAudio status does not belong on the hot path; dropping is deterministic.
                self.frames_dropped += 1
                return
            # The device commonly runs at 44.1/48 kHz stereo. Downmix and
            # linearly resample each fixed capture frame before entering System 1.
            mono = np.asarray(indata, dtype=np.int16).astype(np.float32).mean(axis=1)
            source_positions = np.arange(len(mono), dtype=np.float32)
            target_positions = np.linspace(0, len(mono) - 1, self.config.samples_per_frame)
            data = np.interp(target_positions, source_positions, mono).round().astype(np.int16).tobytes()
            self._sequence += 1
            frame = AudioFrame(
                data=data,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                timestamp=time.monotonic(),
                sequence=self._sequence,
            )
            if self._loop:
                self._loop.call_soon_threadsafe(self._offer_frame, frame)

        try:
            self._stream = sd.InputStream(
                device=self.config.device,
                samplerate=self._capture_sample_rate,
                channels=self._capture_channels,
                dtype="int16",
                blocksize=capture_samples_per_frame,
                callback=callback,
            )
            await asyncio.to_thread(self._stream.start)
        except sd.PortAudioError as error:
            self._closed = True
            raise RuntimeError(
                "Could not open the selected microphone. Run "
                "`python -m ghostpilot.system1.vad_debug --list-devices` and try --device ID."
            ) from error

    def _offer_frame(self, frame: AudioFrame) -> None:
        if self._closed or self._queue.full():
            self.frames_dropped += 1
            return
        self._queue.put_nowait(frame)

    async def frames(self) -> AsyncIterator[AudioFrame]:
        while (frame := await self._queue.get()) is not None:
            yield frame

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stream is not None:
            await asyncio.to_thread(self._stream.stop)
            await asyncio.to_thread(self._stream.close)
            self._stream = None
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(None)
