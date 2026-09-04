import asyncio
import time
import unittest

from ghostpilot.system1.audio import AudioFrame
from ghostpilot.system1.config import EndpointConfig, System1Config
from ghostpilot.system1.endpoint import EndpointState
from ghostpilot.system1.mock_providers import MockDialogueProvider, MockSTTProvider
from ghostpilot.system1.providers import DialogueOutput
from ghostpilot.system1.runtime import System1Runtime
from ghostpilot.system1.state import TurnState
from ghostpilot.system1.testing import FakeAudioInput, FakeVAD
from ghostpilot.system1.vad import VADEvent, VADEventKind


async def wait_until(predicate, timeout: float = 0.75) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true")
        await asyncio.sleep(0.005)


def endpoint_config() -> System1Config:
    return System1Config(endpoint=EndpointConfig(endpoint_timeout_ms=100))


def frame(sequence: int) -> AudioFrame:
    return AudioFrame(
        data=(1_000).to_bytes(2, "little", signed=True) * 320,
        sample_rate=16_000,
        channels=1,
        timestamp=time.monotonic(),
        sequence=sequence,
    )


class MockSTTEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_frames_are_forwarded_to_stt_without_waiting_for_stop(self) -> None:
        audio = FakeAudioInput()
        stt = MockSTTProvider()
        runtime = System1Runtime(
            audio_input=audio,
            vad=FakeVAD([VADEvent(VADEventKind.SPEECH_STARTED)]),
            stt=stt,
            config=endpoint_config(),
        )
        await runtime.start()
        audio.push(frame(1))
        while audio.queue_size:
            await asyncio.sleep(0.005)

        self.assertEqual(stt.audio_frames, [frame(1).data])
        self.assertEqual(runtime.state.turn_state, TurnState.USER_SPEAKING)
        await runtime.close()

    async def test_partial_transcripts_update_and_emit_events_without_commit(self) -> None:
        stt = MockSTTProvider()
        runtime = System1Runtime(stt=stt, config=endpoint_config())
        events = runtime.events.subscribe()
        await runtime.start()
        turn_id = await runtime.on_user_speech_started()
        await stt.emit("open", is_final=False, turn_id=turn_id)
        await stt.emit("open settings", is_final=False, turn_id=turn_id)

        await wait_until(lambda: runtime.transcripts.snapshot().partial == "open settings")
        seen = []
        while not events.empty():
            event = await events.get()
            if event.name == "transcript.partial":
                seen.append(event.text)
        self.assertEqual(seen, ["open", "open settings"])
        self.assertEqual(runtime.state.turn_state, TurnState.USER_SPEAKING)
        await runtime.close()

    async def test_final_transcript_is_preferred_and_emits_event(self) -> None:
        stt = MockSTTProvider()
        runtime = System1Runtime(stt=stt, config=endpoint_config())
        events = runtime.events.subscribe()
        await runtime.start()
        turn_id = await runtime.on_user_speech_started()
        await stt.emit("open settings", is_final=True, turn_id=turn_id)

        await wait_until(lambda: runtime.transcripts.snapshot().final == "open settings")
        seen = []
        while not events.empty():
            seen.append((await events.get()).name)
        self.assertIn("transcript.final", seen)
        self.assertEqual(runtime.transcripts.snapshot().best, "open settings")
        await runtime.close()

    async def test_final_transcript_then_silence_commits_and_starts_dialogue_once(self) -> None:
        stt = MockSTTProvider()
        dialogue = MockDialogueProvider([DialogueOutput("Done.")], delay=0.2)
        runtime = System1Runtime(stt=stt, dialogue=dialogue, config=endpoint_config())
        events = runtime.events.subscribe()
        await runtime.start()
        turn_id = await runtime.on_user_speech_started()
        await stt.emit("open settings", is_final=True, turn_id=turn_id)
        await wait_until(lambda: runtime.transcripts.snapshot().final == "open settings")
        await runtime.on_user_speech_stopped()

        await wait_until(lambda: dialogue.stream_calls == 1)
        seen = []
        while not events.empty():
            seen.append((await events.get()).name)
        self.assertIn("conversation.turn_committed", seen)
        self.assertEqual(runtime.state.committed_transcript, "open settings")
        self.assertEqual(runtime.endpoint_state, EndpointState.COMMITTED)
        await runtime.wait_for_response()
        await runtime.close()

    async def test_pause_then_resume_cancels_endpoint_and_preserves_turn_and_partial(self) -> None:
        stt = MockSTTProvider()
        dialogue = MockDialogueProvider()
        runtime = System1Runtime(stt=stt, dialogue=dialogue, config=endpoint_config())
        await runtime.start()
        turn_id = await runtime.on_user_speech_started()
        buffer = runtime.turn_audio_buffer
        await stt.emit("I think this", is_final=False, turn_id=turn_id)
        await wait_until(lambda: runtime.transcripts.snapshot().partial == "I think this")
        await runtime.on_user_speech_stopped()
        self.assertEqual(runtime.endpoint_state, EndpointState.WAITING)
        await asyncio.sleep(0.03)

        resumed_turn_id = await runtime.on_user_speech_started()
        await asyncio.sleep(0.12)
        self.assertEqual(resumed_turn_id, turn_id)
        self.assertIs(runtime.turn_audio_buffer, buffer)
        self.assertEqual(runtime.transcripts.snapshot().partial, "I think this")
        self.assertEqual(runtime.state.turn_state, TurnState.USER_SPEAKING)
        self.assertEqual(dialogue.stream_calls, 0)
        await runtime.close()

    async def test_resume_then_final_transcript_commits_exactly_once(self) -> None:
        stt = MockSTTProvider()
        dialogue = MockDialogueProvider([DialogueOutput("Okay.")], delay=0.2)
        runtime = System1Runtime(stt=stt, dialogue=dialogue, config=endpoint_config())
        await runtime.start()
        turn_id = await runtime.on_user_speech_started()
        await stt.emit("I think this", is_final=False, turn_id=turn_id)
        await wait_until(lambda: runtime.transcripts.snapshot().partial == "I think this")
        await runtime.on_user_speech_stopped()
        await asyncio.sleep(0.03)
        self.assertEqual(await runtime.on_user_speech_started(), turn_id)
        await stt.emit("I think this is correct", is_final=True, turn_id=turn_id)
        await wait_until(lambda: runtime.transcripts.snapshot().final.endswith("correct"))
        await runtime.on_user_speech_stopped()

        await wait_until(lambda: dialogue.stream_calls == 1)
        self.assertEqual(runtime.state.committed_transcript, "I think this is correct")
        await runtime.wait_for_response()
        await runtime.close()

    async def test_partial_only_does_not_auto_commit(self) -> None:
        stt = MockSTTProvider()
        dialogue = MockDialogueProvider()
        runtime = System1Runtime(stt=stt, dialogue=dialogue, config=endpoint_config())
        await runtime.start()
        turn_id = await runtime.on_user_speech_started()
        await stt.emit("partial only", is_final=False, turn_id=turn_id)
        await wait_until(lambda: runtime.transcripts.snapshot().partial == "partial only")
        await runtime.on_user_speech_stopped()
        await asyncio.sleep(0.14)

        self.assertEqual(runtime.state.turn_state, TurnState.AWAITING_COMMIT)
        self.assertEqual(runtime.endpoint_state, EndpointState.IDLE)
        self.assertEqual(dialogue.stream_calls, 0)
        await runtime.close()

    async def test_stale_endpoint_timer_cannot_commit_after_resume(self) -> None:
        stt = MockSTTProvider()
        dialogue = MockDialogueProvider()
        runtime = System1Runtime(stt=stt, dialogue=dialogue, config=endpoint_config())
        await runtime.start()
        turn_id = await runtime.on_user_speech_started()
        await stt.emit("do not commit", is_final=True, turn_id=turn_id)
        await wait_until(lambda: runtime.transcripts.snapshot().final == "do not commit")
        await runtime.on_user_speech_stopped()
        await asyncio.sleep(0.03)
        await runtime.on_user_speech_started()
        await asyncio.sleep(0.14)

        self.assertEqual(runtime.state.turn_state, TurnState.USER_SPEAKING)
        self.assertEqual(dialogue.stream_calls, 0)
        await runtime.close()

    async def test_stale_stt_event_cannot_mutate_new_turn(self) -> None:
        stt = MockSTTProvider()
        runtime = System1Runtime(stt=stt, config=endpoint_config())
        await runtime.start()
        first_turn = await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("first")
        await runtime.wait_for_response()
        second_turn = await runtime.on_user_speech_started()
        await stt.emit("late first turn", is_final=True, turn_id=first_turn)
        await asyncio.sleep(0.03)

        self.assertNotEqual(first_turn, second_turn)
        self.assertEqual(runtime.transcripts.snapshot().best, "")
        await runtime.close()
