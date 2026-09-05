import asyncio
import json
import unittest

from ghostpilot.system1.adapters.nemotron_stt import NemotronSTTProvider
from ghostpilot.system1.config import NemotronSTTConfig
from ghostpilot.system1.config import EndpointConfig, System1Config
from ghostpilot.system1.runtime import System1Runtime
from ghostpilot.system1.providers import STTEvent, STTServiceEvent


HEALTH = {
    "ok": True,
    "model": "nvidia/nemotron-speech-streaming-en-0.6b",
    "device": "cuda:0",
    "lookahead": 1,
    "sample_rate": 16_000,
    "streaming_latency_ms": 160,
    "warmed_up": True,
}


async def healthy(_url: str) -> dict[str, object]:
    return dict(HEALTH)


class FakeWebSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str | bytes | Exception] = asyncio.Queue()
        self.sent: list[str | bytes] = []
        self.closed = False

    async def send(self, message: str | bytes) -> None:
        if self.closed:
            raise ConnectionError("socket closed")
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        message = await self.incoming.get()
        if isinstance(message, Exception):
            raise message
        return message

    async def close(self) -> None:
        self.closed = True

    def push(self, payload: dict[str, object] | str) -> None:
        self.incoming.put_nowait(json.dumps(payload) if isinstance(payload, dict) else payload)

    def fail(self) -> None:
        self.incoming.put_nowait(ConnectionError("simulated disconnect"))


async def wait_until(predicate, timeout: float = 0.75) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true")
        await asyncio.sleep(0.005)


async def next_matching(provider: NemotronSTTProvider, predicate):
    events = provider.events()
    try:
        while True:
            event = await asyncio.wait_for(anext(events), 0.75)
            if predicate(event):
                return event
    finally:
        await events.aclose()


def test_config(**overrides: object) -> NemotronSTTConfig:
    values: dict[str, object] = {
        "ready_timeout_seconds": 0.2,
        "connect_timeout_seconds": 0.2,
        "reconnect_initial_seconds": 0.01,
        "reconnect_max_seconds": 0.02,
    }
    values.update(overrides)
    return NemotronSTTConfig(**values)  # type: ignore[arg-type]


class NemotronSTTProviderTests(unittest.IsolatedAsyncioTestCase):
    async def make_provider(
        self, sockets: list[FakeWebSocket], **config: object
    ) -> NemotronSTTProvider:
        async def connect(_url: str) -> FakeWebSocket:
            if not sockets:
                raise ConnectionError("no fake socket")
            return sockets.pop(0)

        provider = NemotronSTTProvider(
            test_config(**config), connect_factory=connect, health_fetcher=healthy
        )
        return provider

    async def test_connects_parses_ready_and_validates_health(self) -> None:
        socket = FakeWebSocket()
        socket.push({"type": "ready"})
        provider = await self.make_provider([socket])
        await provider.connect()

        diagnostics = provider.diagnostics()
        self.assertTrue(diagnostics["connected"])
        self.assertTrue(diagnostics["ready"])
        self.assertEqual(diagnostics["health_status"], "healthy")
        self.assertEqual(diagnostics["model"], HEALTH["model"])
        self.assertEqual(json.loads(socket.sent[0]), {"type": "reset"})
        await provider.close()

    async def test_sends_pcm_and_all_control_messages(self) -> None:
        socket = FakeWebSocket()
        socket.push({"type": "ready"})
        provider = await self.make_provider([socket])
        await provider.connect()
        pcm = b"\x01\x00" * 320
        await provider.start_segment("turn-1", 1)
        await provider.send_audio(pcm)
        await wait_until(lambda: pcm in socket.sent)
        await provider.end_segment()
        await wait_until(
            lambda: any(
                isinstance(item, str) and json.loads(item).get("type") == "segment_end"
                for item in socket.sent
            )
        )
        await provider.ping()
        await wait_until(
            lambda: any(
                isinstance(item, str) and json.loads(item).get("type") == "ping"
                for item in socket.sent
            )
        )
        await provider.commit_turn()
        await wait_until(
            lambda: any(
                isinstance(item, str) and json.loads(item).get("type") == "commit"
                for item in socket.sent
            )
        )
        await provider.reset()
        await wait_until(lambda: len(socket.sent) >= 6)

        self.assertIn(pcm, socket.sent)
        controls = [json.loads(item)["type"] for item in socket.sent if isinstance(item, str)]
        self.assertEqual(controls, ["reset", "segment_end", "ping", "commit", "reset"])
        self.assertEqual(provider.diagnostics()["frames_sent"], 1)
        self.assertEqual(provider.diagnostics()["audio_bytes_sent"], len(pcm))
        self.assertEqual(provider.diagnostics()["segment_frames_sent"], 1)
        self.assertEqual(provider.diagnostics()["last_control_sent"], "reset")
        await provider.close()

    async def test_parses_partial_and_final_with_active_context(self) -> None:
        socket = FakeWebSocket()
        socket.push({"type": "ready"})
        provider = await self.make_provider([socket])
        await provider.connect()
        await provider.start_segment("turn-7", 2)
        socket.push({"type": "partial", "text": "open", "segment_id": 2})
        socket.push({"type": "final", "text": "open settings", "segment_id": 2})

        partial = await next_matching(provider, lambda event: isinstance(event, STTEvent))
        final = await next_matching(
            provider, lambda event: isinstance(event, STTEvent) and event.is_final
        )
        self.assertEqual(partial, STTEvent("open", False, "turn-7", 2))
        self.assertEqual(final, STTEvent("open settings", True, "turn-7", 2))
        self.assertEqual(provider.diagnostics()["partial_responses"], 1)
        self.assertEqual(provider.diagnostics()["final_responses"], 1)
        self.assertEqual(provider.diagnostics()["last_response_type"], "final")
        await provider.close()

    async def test_malformed_and_service_error_are_reported_without_crash(self) -> None:
        socket = FakeWebSocket()
        socket.push({"type": "ready"})
        provider = await self.make_provider([socket])
        await provider.connect()
        socket.push("not-json")

        error = await next_matching(
            provider,
            lambda event: isinstance(event, STTServiceEvent) and event.status == "error",
        )
        self.assertIn("malformed", error.detail)
        self.assertTrue(provider.diagnostics()["connected"])
        await provider.close()

    async def test_turn_final_is_acknowledgement_not_a_second_transcript(self) -> None:
        socket = FakeWebSocket()
        socket.push({"type": "ready"})
        provider = await self.make_provider([socket])
        await provider.connect()
        await provider.start_segment("turn-1", 1)
        while not provider._events.empty():
            provider._events.get_nowait()
        socket.push({"type": "turn_final", "text": "must not recommit"})
        await asyncio.sleep(0.03)

        self.assertFalse(
            any(
                isinstance(provider._events.get_nowait(), STTEvent)
                for _ in range(provider._events.qsize())
            )
        )
        await provider.close()

    async def test_bounded_audio_queue_drops_instead_of_growing(self) -> None:
        async def unavailable(_url: str):
            raise ConnectionError("offline")

        provider = NemotronSTTProvider(
            test_config(send_queue_size=2),
            connect_factory=unavailable,
            health_fetcher=healthy,
        )
        await provider.connect()
        for _ in range(10):
            await provider.send_audio(b"frame")

        self.assertLessEqual(provider.diagnostics()["send_queue_depth"], 2)
        self.assertEqual(provider.diagnostics()["stt_dropped_frames"], 8)
        await provider.close()

    async def test_unavailable_service_does_not_crash_system1_startup(self) -> None:
        async def unavailable(_url: str):
            raise ConnectionError("service unavailable")

        provider = NemotronSTTProvider(
            test_config(ready_timeout_seconds=0.05),
            connect_factory=unavailable,
            health_fetcher=healthy,
        )
        runtime = System1Runtime(stt=provider, config=System1Config(stt_provider="nemotron"))
        events = runtime.events.subscribe()
        await runtime.start()
        await wait_until(lambda: not events.empty())
        await asyncio.sleep(0.02)
        seen = []
        while not events.empty():
            seen.append((await events.get()).name)

        self.assertTrue(runtime._stt_task and not runtime._stt_task.done())
        self.assertIn("system.provider_failed", seen)
        await runtime.close()

    async def test_disconnect_reconnects_and_stale_generation_is_ignored(self) -> None:
        first, second = FakeWebSocket(), FakeWebSocket()
        first.push({"type": "ready"})
        second.push({"type": "ready"})
        provider = await self.make_provider([first, second])
        await provider.connect()
        await provider.start_segment("turn-1", 1)
        first.fail()
        await wait_until(lambda: provider.diagnostics()["connection_generation"] == 2)
        await wait_until(lambda: provider.diagnostics()["ready"] is True)

        await provider._handle_message(
            json.dumps({"type": "final", "text": "stale", "segment_id": 1}), 1
        )
        self.assertIn("stale connection", provider.diagnostics()["last_ignored_reason"])
        second.push({"type": "partial", "text": "fresh", "segment_id": 1})
        fresh = await next_matching(
            provider,
            lambda event: isinstance(event, STTEvent) and event.text == "fresh",
        )
        self.assertEqual(fresh.turn_id, "turn-1")
        self.assertGreaterEqual(provider.diagnostics()["reconnect_count"], 1)
        await provider.close()

    async def test_wrong_segment_response_exposes_rejection_reason(self) -> None:
        socket = FakeWebSocket()
        socket.push({"type": "ready"})
        provider = await self.make_provider([socket])
        await provider.connect()
        await provider.start_segment("turn-1", 3)
        socket.push({"type": "final", "text": "old result", "segment_id": 2})
        await wait_until(lambda: provider.diagnostics()["ignored_responses"] == 1)

        diagnostics = provider.diagnostics()
        self.assertEqual(diagnostics["final_responses"], 0)
        self.assertIn("segment 2 != active 3", diagnostics["last_ignored_reason"])
        self.assertTrue(any("response ignored" in item for item in diagnostics["trace"]))
        await provider.close()

    async def test_runtime_survives_reconnect_and_accepts_fresh_transcript(self) -> None:
        first, second = FakeWebSocket(), FakeWebSocket()
        first.push({"type": "ready"})
        second.push({"type": "ready"})
        provider = await self.make_provider([first, second])
        runtime = System1Runtime(
            stt=provider,
            config=System1Config(
                stt_provider="nemotron",
                endpoint=EndpointConfig(endpoint_timeout_ms=100),
            ),
        )
        await runtime.start()
        turn_id = await runtime.on_user_speech_started()
        first.fail()
        await wait_until(lambda: provider.diagnostics()["connection_generation"] == 2)
        await wait_until(lambda: provider.diagnostics()["ready"] is True)
        second.push({"type": "partial", "text": "still alive", "segment_id": 1})

        await wait_until(lambda: runtime.transcripts.snapshot().partial == "still alive")
        self.assertEqual(runtime.state.current_turn, turn_id)
        self.assertFalse(runtime._stt_task.done())
        await runtime.close()
