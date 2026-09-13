import asyncio
from collections.abc import AsyncIterator
import unittest

from ghostpilot.system1.mock_providers import MockPlayback, MockSTTProvider, MockTTSProvider
from ghostpilot.system1.providers import DialogueOutput
from ghostpilot.system1.runtime import System1Runtime
from ghostpilot.system1.state import TurnState


class ControlledDialogue:
    def __init__(self, responses: list[list[str]]) -> None:
        self.responses = responses
        self.stream_calls = 0
        self.cancel_calls = 0
        self.started = asyncio.Event()
        self.allow_old = asyncio.Event()

    async def stream(self, transcript: str) -> AsyncIterator[DialogueOutput]:
        call_index = self.stream_calls
        self.stream_calls += 1
        self.started.set()
        if call_index == 0 and len(self.responses) > 1:
            # Mimic a transport which can still produce a chunk after cancellation.
            await self.allow_old.wait()
        for text in self.responses[call_index]:
            yield DialogueOutput(text)

    async def cancel(self) -> None:
        self.cancel_calls += 1

    def diagnostics(self) -> dict[str, object]:
        return {"provider": "controlled", "request_active": False, "live_text": ""}


class DialogueRuntimeTests(unittest.IsolatedAsyncioTestCase):
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
        self.assertEqual(dialogue.cancel_calls, 1)
        self.assertEqual(runtime.state.turn_state, TurnState.USER_SPEAKING)
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("second")
        dialogue.allow_old.set()
        await runtime.wait_for_response()
        await asyncio.sleep(0)  # Let the deliberately late first stream observe ownership.

        self.assertEqual([chunk.data for chunk in playback.played], [b"New response."])
        await runtime.close()
