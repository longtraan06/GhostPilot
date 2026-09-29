import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
import time
import unittest

from ghostpilot.system1.mock_providers import (
    MockDialogueProvider,
    MockPlayback,
    MockSTTProvider,
    MockTTSProvider,
)
from ghostpilot.system1.providers import DialogueMessage, DialogueOutput, DialogueProviderError
from ghostpilot.system1.runtime import System1Runtime
from ghostpilot.system1.state import TurnState


@dataclass
class ImmediateDialogueCleanup:
    async def close(self) -> None:
        return None


class ControlledDialogue:
    def __init__(self, responses: list[list[str]]) -> None:
        self.responses = responses
        self.stream_calls = 0
        self.cancel_calls = 0
        self.started = asyncio.Event()
        self.allow_old = asyncio.Event()
        self._active = False

    async def stream(self, messages: Sequence[DialogueMessage]) -> AsyncIterator[DialogueOutput]:
        call_index = self.stream_calls
        self.stream_calls += 1
        self._active = True
        self.started.set()
        if call_index == 0 and len(self.responses) > 1:
            # Mimic a transport which can still produce a chunk after cancellation.
            await self.allow_old.wait()
        for text in self.responses[call_index]:
            yield DialogueOutput(text)

    def invalidate_active(self):
        if not self._active:
            return None
        self._active = False
        self.cancel_calls += 1
        return ImmediateDialogueCleanup()

    async def cancel(self) -> None:
        self.invalidate_active()

    def diagnostics(self) -> dict[str, object]:
        return {"provider": "controlled", "request_active": False, "live_text": ""}


class SlowCancelDialogue:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.cancel_calls = 0
        self._active = False

    async def stream(self, messages: Sequence[DialogueMessage]) -> AsyncIterator[DialogueOutput]:
        self.started.set()
        self._active = True
        yield DialogueOutput("Still speaking.")
        await asyncio.Event().wait()

    def invalidate_active(self):
        if not self._active:
            return None
        self._active = False
        return SlowDialogueCleanup(self)

    async def cancel(self) -> None:
        cleanup = self.invalidate_active()
        if cleanup is not None:
            await cleanup.close()

    async def _finish_cancel(self) -> None:
        self.cancel_calls += 1
        await asyncio.sleep(0.2)
        self.cancelled.set()

    def diagnostics(self) -> dict[str, object]:
        return {"provider": "slow-dialogue", "last_error": ""}


@dataclass
class SlowDialogueCleanup:
    provider: SlowCancelDialogue

    async def close(self) -> None:
        await self.provider._finish_cancel()


class BlockingDialogue:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancel_calls = 0

    async def stream(self, messages: Sequence[DialogueMessage]) -> AsyncIterator[DialogueOutput]:
        self.started.set()
        await asyncio.Event().wait()
        yield DialogueOutput("unreachable")

    async def cancel(self) -> None:
        self.cancel_calls += 1

    def invalidate_active(self):
        return ImmediateDialogueCleanup()

    def diagnostics(self) -> dict[str, object]:
        return {"provider": "blocking-dialogue", "last_error": ""}


class ScriptedDialogue:
    def __init__(self, scripts: list[list[object]]) -> None:
        self.scripts = scripts
        self.stream_calls = 0
        self.cancel_calls = 0

    async def stream(self, messages: Sequence[DialogueMessage]) -> AsyncIterator[DialogueOutput]:
        script = self.scripts[self.stream_calls]
        self.stream_calls += 1
        for item in script:
            if isinstance(item, BaseException):
                raise item
            yield DialogueOutput(str(item))

    async def cancel(self) -> None:
        self.cancel_calls += 1

    def invalidate_active(self):
        return ImmediateDialogueCleanup()

    def diagnostics(self) -> dict[str, object]:
        return {"provider": "scripted-dialogue", "last_error": ""}


class PausedCleanupDialogue:
    """Models a request-specific old close that completes after request two starts."""

    def __init__(self) -> None:
        self.generation = 0
        self.active_generation: int | None = None
        self.first_started = asyncio.Event()
        self.second_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.release_second = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        self.closed_generations: list[int] = []
        self.requests: list[tuple[DialogueMessage, ...]] = []

    async def stream(self, messages: Sequence[DialogueMessage]) -> AsyncIterator[DialogueOutput]:
        self.requests.append(tuple(messages))
        self.generation += 1
        generation = self.generation
        self.active_generation = generation
        if generation == 1:
            self.first_started.set()
            yield DialogueOutput("Old response.")
            await self.release_first.wait()
            if generation == self.active_generation:
                yield DialogueOutput("Late old response.")
            return
        self.second_started.set()
        await self.release_second.wait()
        if generation == self.active_generation:
            yield DialogueOutput("New response.")

    def invalidate_active(self):
        generation = self.active_generation
        if generation is None:
            return None
        self.active_generation = None
        return PausedDialogueCleanup(self, generation)

    async def cancel(self) -> None:
        cleanup = self.invalidate_active()
        if cleanup is not None:
            await cleanup.close()

    def diagnostics(self) -> dict[str, object]:
        return {"provider": "paused-cleanup", "last_error": ""}


@dataclass
class PausedDialogueCleanup:
    provider: PausedCleanupDialogue
    generation: int

    async def close(self) -> None:
        self.provider.cleanup_started.set()
        await self.provider.release_cleanup.wait()
        self.provider.closed_generations.append(self.generation)


class DialogueRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_exchanges_build_short_term_context_and_join_raw_chunks(self) -> None:
        dialogue = MockDialogueProvider([DialogueOutput("Python"), DialogueOutput(" is useful.")])
        runtime = System1Runtime(dialogue=dialogue)
        await runtime.start()

        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("My favourite language is Python.")
        await runtime.wait_for_response()
        self.assertEqual(runtime.turns.history.exchange_count, 1)
        self.assertEqual(
            runtime.turns.history.exchanges()[0].assistant, "Python is useful."
        )
        self.assertEqual(
            [(message.role, message.content) for message in dialogue.requests[0]],
            [
                ("system", runtime.config.dialogue_context.system_prompt),
                ("user", "My favourite language is Python."),
            ],
        )

        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("What language did I mention?")
        await runtime.wait_for_response()
        self.assertEqual(
            [(message.role, message.content) for message in dialogue.requests[1]],
            [
                ("system", runtime.config.dialogue_context.system_prompt),
                ("user", "My favourite language is Python."),
                ("assistant", "Python is useful."),
                ("user", "What language did I mention?"),
            ],
        )
        self.assertEqual(runtime.turns.history.exchange_count, 2)
        await runtime.close()

    async def test_explicit_history_clear_and_stt_reset_do_not_share_lifecycle(self) -> None:
        dialogue = MockDialogueProvider([DialogueOutput("Done.")])
        stt = MockSTTProvider()
        runtime = System1Runtime(stt=stt, dialogue=dialogue)
        await runtime.start()
        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("remember this")
        await runtime.wait_for_response()
        self.assertEqual(runtime.turns.history.exchange_count, 1)

        await stt.reset()
        self.assertEqual(runtime.turns.history.exchange_count, 1)
        runtime.clear_dialogue_history()
        self.assertEqual(runtime.turns.history.exchange_count, 0)

        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("fresh turn")
        await runtime.wait_for_response()
        self.assertEqual(
            [(message.role, message.content) for message in dialogue.requests[-1]],
            [("system", runtime.config.dialogue_context.system_prompt), ("user", "fresh turn")],
        )
        await runtime.close()

    async def test_late_old_barge_in_cleanup_cannot_cancel_new_dialogue_request(self) -> None:
        dialogue = PausedCleanupDialogue()
        playback = MockPlayback()
        runtime = System1Runtime(dialogue=dialogue, playback=playback)
        await runtime.start()
        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("first")
        await dialogue.first_started.wait()
        await asyncio.sleep(0)

        await runtime.on_user_speech_started()
        await dialogue.cleanup_started.wait()
        self.assertEqual(runtime.turns.history.exchange_count, 0)
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("second")
        await dialogue.second_started.wait()
        self.assertEqual(dialogue.active_generation, 2)
        self.assertEqual(
            [(message.role, message.content) for message in dialogue.requests[1]],
            [("system", runtime.config.dialogue_context.system_prompt), ("user", "second")],
        )

        dialogue.release_cleanup.set()
        await runtime.interruption.wait_for_cancellations()
        self.assertEqual(dialogue.closed_generations, [1])
        self.assertEqual(dialogue.active_generation, 2)

        dialogue.release_second.set()
        await runtime.wait_for_response()
        dialogue.release_first.set()
        await asyncio.sleep(0)
        self.assertEqual([audio.data for audio in playback.played], [b"Old response.", b"New response."])
        self.assertEqual(
            [(exchange.user, exchange.assistant) for exchange in runtime.turns.history.exchanges()],
            [("second", "New response.")],
        )
        await runtime.close()

    async def test_barge_in_does_not_wait_for_slow_provider_cancellation(self) -> None:
        dialogue = SlowCancelDialogue()
        runtime = System1Runtime(dialogue=dialogue, tts=MockTTSProvider(delay=1))
        await runtime.start()
        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("first")
        await dialogue.started.wait()
        await asyncio.sleep(0)

        started_at = time.monotonic()
        await runtime.on_user_speech_started()
        elapsed = time.monotonic() - started_at

        self.assertLess(elapsed, 0.1)
        self.assertEqual(runtime.state.turn_state, TurnState.USER_SPEAKING)
        await asyncio.wait_for(dialogue.cancelled.wait(), timeout=1)
        await runtime.close()

    async def test_partial_stt_never_starts_dialogue(self) -> None:
        stt = MockSTTProvider()
        dialogue = ControlledDialogue([["Should not run."]])
        runtime = System1Runtime(stt=stt, dialogue=dialogue)
        await runtime.start()

        turn_id = await runtime.on_user_speech_started()
        await stt.emit("partial only", is_final=False, turn_id=turn_id, segment_id=1)
        await asyncio.sleep(0)

        self.assertEqual(dialogue.stream_calls, 0)
        await runtime.close()

    async def test_dialogue_starts_only_after_explicit_turn_commit(self) -> None:
        dialogue = ControlledDialogue([["Ready."]])
        runtime = System1Runtime(dialogue=dialogue)
        await runtime.start()

        await runtime.on_user_speech_started()
        self.assertEqual(dialogue.stream_calls, 0)
        await runtime.on_user_speech_stopped()
        self.assertEqual(runtime.state.turn_state, TurnState.AWAITING_COMMIT)
        self.assertEqual(dialogue.stream_calls, 0)

        await runtime.commit_turn("committed transcript")
        await runtime.wait_for_response()
        self.assertEqual(dialogue.stream_calls, 1)
        await runtime.close()

    async def test_incremental_chunks_are_speech_segmented_without_waiting_for_full_reply(self) -> None:
        dialogue = ControlledDialogue([["First. ", "Second."]])
        playback = MockPlayback()
        runtime = System1Runtime(dialogue=dialogue, tts=MockTTSProvider(), playback=playback)
        await runtime.start()

        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("say two sentences")
        await runtime.wait_for_response()

        self.assertEqual([chunk.data for chunk in playback.played], [b"First.", b"Second."])
        await runtime.close()

    async def test_barge_in_cancels_dialogue_and_late_old_output_cannot_speak_new_turn(self) -> None:
        dialogue = ControlledDialogue([["Old response."], ["New response."]])
        playback = MockPlayback()
        runtime = System1Runtime(dialogue=dialogue, tts=MockTTSProvider(), playback=playback)
        await runtime.start()

        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("first")
        await dialogue.started.wait()

        await runtime.on_user_speech_started()
        await runtime.interruption.wait_for_cancellations()
        self.assertEqual(dialogue.cancel_calls, 1)
        self.assertEqual(runtime.state.turn_state, TurnState.USER_SPEAKING)
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("second")
        dialogue.allow_old.set()
        await runtime.wait_for_response()
        await asyncio.sleep(0)  # Let the deliberately late first stream observe ownership.

        self.assertEqual([chunk.data for chunk in playback.played], [b"New response."])
        await runtime.close()

    async def test_shutdown_cancels_and_awaits_active_response_task(self) -> None:
        dialogue = BlockingDialogue()
        runtime = System1Runtime(dialogue=dialogue)
        await runtime.start()
        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("wait")
        await dialogue.started.wait()

        await asyncio.wait_for(runtime.close(), timeout=1)
        self.assertGreaterEqual(dialogue.cancel_calls, 1)
        self.assertIsNone(runtime.turns._response_task)
        self.assertFalse(runtime.turns._response_tasks)

    async def test_shutdown_is_safe_while_barge_in_cleanup_is_running_and_when_idle(self) -> None:
        dialogue = SlowCancelDialogue()
        runtime = System1Runtime(dialogue=dialogue, tts=MockTTSProvider(delay=1))
        await runtime.start()
        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("first")
        await dialogue.started.wait()
        await runtime.on_user_speech_started()

        await asyncio.wait_for(runtime.close(), timeout=1)
        self.assertGreaterEqual(dialogue.cancel_calls, 1)

        idle = System1Runtime()
        await idle.start()
        await idle.close()
        await idle.close()

    async def test_dialogue_failure_emits_provider_event_recovers_and_next_turn_works(self) -> None:
        dialogue = ScriptedDialogue(
            [[DialogueProviderError("connection refused")], ["Recovered."]]
        )
        runtime = System1Runtime(dialogue=dialogue)
        events = runtime.events.subscribe()
        await runtime.start()
        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("first")
        await runtime.wait_for_response()

        seen = []
        while not events.empty():
            seen.append(await events.get())
        failure = next(event for event in seen if event.name == "system.provider_failed")
        self.assertEqual(failure.provider, "scripted-dialogue")
        self.assertIn("connection refused", failure.detail)
        self.assertEqual(runtime.state.turn_state, TurnState.LISTENING)
        self.assertEqual(runtime.turns.history.exchange_count, 0)

        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("second")
        await runtime.wait_for_response()
        self.assertEqual(dialogue.stream_calls, 2)
        self.assertEqual(runtime.state.turn_state, TurnState.LISTENING)
        await runtime.close()

    async def test_mid_stream_dialogue_failure_keeps_prior_content_and_recovers(self) -> None:
        dialogue = ScriptedDialogue([["Hello.", DialogueProviderError("stream broke")]])
        playback = MockPlayback()
        runtime = System1Runtime(dialogue=dialogue, playback=playback)
        events = runtime.events.subscribe()
        await runtime.start()
        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("first")
        await runtime.wait_for_response()

        self.assertEqual([audio.data for audio in playback.played], [b"Hello."])
        seen = []
        while not events.empty():
            seen.append((await events.get()).name)
        self.assertIn("system.provider_failed", seen)
        self.assertEqual(runtime.state.turn_state, TurnState.LISTENING)
        self.assertEqual(runtime.turns.history.exchange_count, 0)
        await runtime.close()

    async def test_empty_completed_response_is_not_added_to_history(self) -> None:
        runtime = System1Runtime(dialogue=ScriptedDialogue([[]]))
        await runtime.start()
        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("no text should be stored")
        await runtime.wait_for_response()

        self.assertEqual(runtime.turns.history.exchange_count, 0)
        await runtime.close()
