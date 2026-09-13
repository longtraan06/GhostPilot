import asyncio
from types import SimpleNamespace
import unittest

from ghostpilot.system1.adapters.openai_compatible_dialogue import (
    OpenAICompatibleDialogueProvider,
)
from ghostpilot.system1.config import OpenAICompatibleDialogueConfig
from ghostpilot.system1.providers import DialogueProviderError


def chunk(content=None, *, finish_reason=None):
    delta = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class FakeStream:
    def __init__(self, items, *, delay=0.0):
        self.items = list(items)
        self.delay = delay
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        if self.delay:
            await asyncio.sleep(self.delay)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self):
        self.closed += 1


class LateStream:
    """Deliberately yields after close to model a late transport chunk."""

    def __init__(self, value):
        self.value = value
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = 0
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._sent:
            raise StopAsyncIteration
        self.entered.set()
        await self.release.wait()
        self._sent = True
        return chunk(self.value)

    async def close(self):
        self.closed += 1


class FakeCompletions:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeClient:
    def __init__(self, results):
        self.completions = FakeCompletions(results)
        self.chat = SimpleNamespace(completions=self.completions)


class SlowCreateClient:
    def __init__(self, stream):
        self.stream = stream
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **_kwargs):
        self.entered.set()
        await self.release.wait()
        return self.stream


def provider_for(*results):
    client = FakeClient(results)
    provider = OpenAICompatibleDialogueProvider(
        OpenAICompatibleDialogueConfig(model="served-model"), client=client
    )
    return provider, client


async def collect(provider, transcript="hello"):
    return [output.text async for output in provider.stream(transcript)]


class OpenAICompatibleDialogueProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_non_empty_deltas_and_closes_normally(self):
        stream = FakeStream([chunk("Hello"), chunk(", "), chunk("world", finish_reason="stop")])
        provider, client = provider_for(stream)

        self.assertEqual(await collect(provider, "How are you?"), ["Hello", ", ", "world"])
        self.assertEqual(stream.closed, 1)
        self.assertEqual(client.completions.calls[0]["model"], "served-model")
        self.assertEqual(
            client.completions.calls[0]["messages"][-1], {"role": "user", "content": "How are you?"}
        )
        diagnostics = provider.diagnostics()
        self.assertEqual(diagnostics["requests_completed"], 1)
        self.assertEqual(diagnostics["last_finish_reason"], "stop")

    async def test_ignores_empty_and_malformed_chunks_and_measures_ttft_on_content(self):
        stream = FakeStream(
            [SimpleNamespace(choices=[]), chunk(None), chunk(""), {"choices": [{}]}, chunk("Useful")],
            delay=0.002,
        )
        provider, _ = provider_for(stream)

        self.assertEqual(await collect(provider), ["Useful"])
        diagnostics = provider.diagnostics()
        self.assertEqual(diagnostics["chunks_received"], 1)
        self.assertEqual(diagnostics["characters_received"], len("Useful"))
        self.assertIsNotNone(diagnostics["last_ttft_ms"])
        self.assertGreater(diagnostics["last_ttft_ms"], 0)

    async def test_cancel_is_idempotent_and_suppresses_late_content(self):
        stream = LateStream("late")
        provider, _ = provider_for(stream)
        task = asyncio.create_task(collect(provider))
        await stream.entered.wait()

        await provider.cancel()
        await provider.cancel()
        await provider.cancel()
        stream.release.set()
        self.assertEqual(await task, [])
        diagnostics = provider.diagnostics()
        self.assertEqual(stream.closed, 2)  # cancel plus the request's final cleanup
        self.assertEqual(diagnostics["requests_cancelled"], 1)
        self.assertFalse(diagnostics["request_active"])

    async def test_cancel_during_connection_establishment_closes_eventual_stream(self):
        stream = FakeStream([chunk("too late")])
        client = SlowCreateClient(stream)
        provider = OpenAICompatibleDialogueProvider(
            OpenAICompatibleDialogueConfig(model="served-model"), client=client
        )
        task = asyncio.create_task(collect(provider))
        await client.entered.wait()
        await provider.cancel()
        client.release.set()

        self.assertEqual(await task, [])
        self.assertEqual(stream.closed, 1)
        self.assertEqual(provider.diagnostics()["requests_cancelled"], 1)

    async def test_old_generation_and_cleanup_cannot_disturb_new_request(self):
        old_stream = LateStream("old response")
        new_stream = LateStream("new response")
        provider, _ = provider_for(old_stream, new_stream)
        old_task = asyncio.create_task(collect(provider, "old"))
        await old_stream.entered.wait()
        await provider.cancel()

        new_task = asyncio.create_task(collect(provider, "new"))
        await new_stream.entered.wait()
        old_stream.release.set()
        self.assertEqual(await old_task, [])
        self.assertTrue(provider.diagnostics()["request_active"])

        new_stream.release.set()
        self.assertEqual(await new_task, ["new response"])
        self.assertFalse(provider.diagnostics()["request_active"])
        self.assertEqual(provider.diagnostics()["live_text"], "new response")

    async def test_recovers_after_connection_or_stream_error(self):
        good_stream = FakeStream([chunk("Recovered.")])
        provider, _ = provider_for(ConnectionError("server unavailable"), good_stream)

        with self.assertRaises(DialogueProviderError):
            await collect(provider, "first")
        self.assertEqual(provider.diagnostics()["requests_failed"], 1)
        self.assertIn("ConnectionError", provider.diagnostics()["last_error"])
        self.assertEqual(await collect(provider, "second"), ["Recovered."])
        self.assertEqual(provider.diagnostics()["requests_completed"], 1)

    async def test_base_url_accepts_server_root_and_model_is_explicit(self):
        stream = FakeStream([chunk("ok")])
        provider = OpenAICompatibleDialogueProvider(
            OpenAICompatibleDialogueConfig(base_url="http://localhost:3308/", model="configured"),
            client=FakeClient([stream]),
        )
        await collect(provider)
        self.assertEqual(provider.diagnostics()["base_url"], "http://localhost:3308/v1")

        missing_model = OpenAICompatibleDialogueProvider(
            OpenAICompatibleDialogueConfig(model=""), client=FakeClient([])
        )
        with self.assertRaises(DialogueProviderError):
            await collect(missing_model)
        self.assertIn("GHOSTPILOT_DIALOGUE_MODEL", missing_model.diagnostics()["last_error"])

    async def test_passes_configured_server_extension_without_coupling_turn_manager(self):
        stream = FakeStream([chunk("Direct response.")])
        client = FakeClient([stream])
        provider = OpenAICompatibleDialogueProvider(
            OpenAICompatibleDialogueConfig(
                model="served-model",
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
            client=client,
        )

        self.assertEqual(await collect(provider), ["Direct response."])
        self.assertEqual(
            client.completions.calls[0]["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
