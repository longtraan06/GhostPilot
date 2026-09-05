import asyncio
import time
import unittest

from ghostpilot.system1.audio import AudioFrame
from ghostpilot.system1.config import AudioConfig, System1Config, VADConfig
from ghostpilot.system1.mock_providers import (
    MockDialogueProvider,
    MockPlayback,
    MockSTTProvider,
    MockTTSProvider,
)
from ghostpilot.system1.providers import DialogueOutput
from ghostpilot.system1.runtime import System1Runtime
from ghostpilot.system1.state import TurnState
from ghostpilot.system1.testing import FakeAudioInput, FakeVAD
from ghostpilot.system1.vad import VADEvent, VADEventKind


def frame(sequence: int, amplitude: int = 0) -> AudioFrame:
    return AudioFrame(
        data=(amplitude.to_bytes(2, "little", signed=True) * 320),
        sample_rate=16_000,
        channels=1,
        timestamp=time.monotonic(),
        sequence=sequence,
    )


async def wait_until(predicate, timeout: float = 0.5) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true")
        await asyncio.sleep(0.005)


class AudioVADRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_stt_is_speech_gated_and_receives_only_bounded_pre_roll(self) -> None:
        audio = FakeAudioInput()
        stt = MockSTTProvider()
        vad = FakeVAD(
            [
                None,
                None,
                None,
                None,
                None,
                VADEvent(VADEventKind.SPEECH_STARTED),
                None,
                VADEvent(VADEventKind.SPEECH_STOPPED),
            ]
        )
        runtime = System1Runtime(
            audio_input=audio,
            vad=vad,
            stt=stt,
            config=System1Config(audio=AudioConfig(pre_roll_ms=60)),
        )
        await runtime.start()
        for sequence in range(1, 9):
            audio.push(frame(sequence, sequence))

        await wait_until(
            lambda: audio.queue_size == 0
            and runtime.state.turn_state is TurnState.AWAITING_COMMIT
        )
        sent_amplitudes = [
            int.from_bytes(item[:2], "little", signed=True) for item in stt.audio_frames
        ]
        self.assertEqual(sent_amplitudes, [4, 5, 6, 7, 8])
        self.assertEqual(stt.segment_end_calls, 1)
        await runtime.close()

    async def test_pre_roll_is_prepended_to_new_turn_in_sequence_order(self) -> None:
        audio = FakeAudioInput()
        vad = FakeVAD([None, None, None, VADEvent(VADEventKind.SPEECH_STARTED, 0.9)])
        runtime = System1Runtime(
            audio_input=audio,
            vad=vad,
            config=System1Config(audio=AudioConfig(pre_roll_ms=60)),
        )
        await runtime.start()
        for sequence in range(1, 5):
            audio.push(frame(sequence, 5_000 if sequence == 4 else 0))

        await wait_until(lambda: runtime.state.turn_state is TurnState.USER_SPEAKING)
        self.assertEqual(
            [item.sequence for item in runtime.turn_audio_buffer.snapshot()], [2, 3, 4]
        )
        self.assertEqual(runtime.debug_snapshot()["pre_roll_frame_count"], 3)
        await runtime.close()

    async def test_speech_start_trigger_frame_is_not_duplicated(self) -> None:
        audio = FakeAudioInput()
        vad = FakeVAD([None, None, None, VADEvent(VADEventKind.SPEECH_STARTED, 0.9)])
        runtime = System1Runtime(
            audio_input=audio,
            vad=vad,
            config=System1Config(audio=AudioConfig(pre_roll_ms=60)),
        )
        await runtime.start()
        for sequence in range(1, 5):
            audio.push(frame(sequence, 5_000 if sequence == 4 else 0))

        await wait_until(lambda: runtime.state.turn_state is TurnState.USER_SPEAKING)
        sequences = [item.sequence for item in runtime.turn_audio_buffer.snapshot()]
        self.assertEqual(sequences.count(4), 1)
        await runtime.close()

    async def test_old_pre_roll_frames_are_evicted(self) -> None:
        audio = FakeAudioInput()
        runtime = System1Runtime(
            audio_input=audio,
            vad=FakeVAD([None] * 5),
            config=System1Config(audio=AudioConfig(pre_roll_ms=60)),
        )
        await runtime.start()
        for sequence in range(1, 6):
            audio.push(frame(sequence))

        await wait_until(lambda: audio.queue_size == 0)
        self.assertEqual([item.sequence for item in runtime.pre_roll_buffer.snapshot()], [3, 4, 5])
        self.assertEqual(runtime.pre_roll_buffer.frame_count, 3)
        await runtime.close()

    async def test_pre_roll_and_multiple_segments_remain_in_one_turn(self) -> None:
        audio = FakeAudioInput()
        vad = FakeVAD(
            [
                None,
                None,
                None,
                VADEvent(VADEventKind.SPEECH_STARTED),
                VADEvent(VADEventKind.SPEECH_STOPPED),
                None,
                VADEvent(VADEventKind.SPEECH_STARTED),
                VADEvent(VADEventKind.SPEECH_STOPPED),
            ]
        )
        dialogue, tts = MockDialogueProvider(), MockTTSProvider()
        runtime = System1Runtime(
            audio_input=audio,
            vad=vad,
            dialogue=dialogue,
            tts=tts,
            config=System1Config(audio=AudioConfig(pre_roll_ms=60)),
        )
        await runtime.start()
        for sequence in range(1, 9):
            audio.push(frame(sequence, 5_000 if sequence in {4, 7} else 0))

        await wait_until(
            lambda: audio.queue_size == 0 and runtime.state.turn_state is TurnState.AWAITING_COMMIT
        )
        self.assertEqual(runtime.turn_audio_buffer.turn_id, "turn-1")
        self.assertEqual(
            [item.sequence for item in runtime.turn_audio_buffer.snapshot()],
            [2, 3, 4, 5, 6, 7, 8],
        )
        self.assertEqual(dialogue.stream_calls, 0)
        self.assertEqual(tts.stream_calls, 0)
        await runtime.close()

    async def test_resumed_segment_pre_roll_excludes_previous_segment_audio(self) -> None:
        audio = FakeAudioInput()
        stt = MockSTTProvider()
        vad = FakeVAD(
            [
                None,
                None,
                None,
                VADEvent(VADEventKind.SPEECH_STARTED),
                None,
                VADEvent(VADEventKind.SPEECH_STOPPED),
                None,
                VADEvent(VADEventKind.SPEECH_STARTED),
                None,
                VADEvent(VADEventKind.SPEECH_STOPPED),
            ]
        )
        runtime = System1Runtime(
            audio_input=audio,
            vad=vad,
            stt=stt,
            config=System1Config(audio=AudioConfig(pre_roll_ms=60)),
        )
        await runtime.start()
        for sequence in range(1, 11):
            audio.push(frame(sequence, sequence))

        await wait_until(
            lambda: audio.queue_size == 0
            and runtime.state.turn_state is TurnState.AWAITING_COMMIT
        )
        sent = [
            int.from_bytes(item[:2], "little", signed=True)
            for item in stt.audio_frames
        ]
        self.assertEqual(sent, [2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(sent.count(6), 1)
        self.assertIn(8, sent)  # resumed speech trigger frame is protected
        self.assertEqual(stt.segment_starts, [("turn-1", 1), ("turn-1", 2)])
        await runtime.close()

    async def test_speech_start_from_audio_path_emits_event_and_enters_user_speaking(self) -> None:
        audio = FakeAudioInput()
        vad = FakeVAD([None, VADEvent(VADEventKind.SPEECH_STARTED, 0.9)])
        runtime = System1Runtime(audio_input=audio, vad=vad)
        events = runtime.events.subscribe()
        await runtime.start()
        audio.push(frame(1))
        audio.push(frame(2, 5_000))

        await wait_until(lambda: runtime.state.turn_state is TurnState.USER_SPEAKING)
        seen = []
        while not events.empty():
            seen.append((await events.get()).name)
        self.assertIn("audio.speech_started", seen)
        self.assertEqual(
            [item.sequence for item in runtime.turn_audio_buffer.snapshot()], [1, 2]
        )
        await runtime.close()

    async def test_speech_stop_from_audio_path_does_not_commit_or_start_providers(self) -> None:
        audio = FakeAudioInput()
        vad = FakeVAD(
            [
                VADEvent(VADEventKind.SPEECH_STARTED, 0.9),
                VADEvent(VADEventKind.SPEECH_STOPPED, 0.1),
            ]
        )
        dialogue, tts = MockDialogueProvider(), MockTTSProvider()
        runtime = System1Runtime(audio_input=audio, vad=vad, dialogue=dialogue, tts=tts)
        events = runtime.events.subscribe()
        await runtime.start()
        audio.push(frame(1, 5_000))
        audio.push(frame(2))

        await wait_until(
            lambda: audio.queue_size == 0 and runtime.state.turn_state is TurnState.AWAITING_COMMIT
        )
        seen = []
        while not events.empty():
            seen.append((await events.get()).name)
        self.assertIn("audio.speech_stopped", seen)
        self.assertNotIn("conversation.turn_committed", seen)
        self.assertEqual(dialogue.stream_calls, 0)
        self.assertEqual(tts.stream_calls, 0)
        await runtime.close()

    async def test_audio_path_barge_in_is_local_first(self) -> None:
        audio = FakeAudioInput()
        vad = FakeVAD([VADEvent(VADEventKind.SPEECH_STARTED, 0.95)])
        dialogue = MockDialogueProvider([DialogueOutput("Still speaking.")], delay=0.02)
        tts = MockTTSProvider(delay=0.2)
        playback = MockPlayback()
        runtime = System1Runtime(
            audio_input=audio, vad=vad, dialogue=dialogue, tts=tts, playback=playback
        )
        await runtime.start()
        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("first turn")
        await asyncio.sleep(0.04)
        self.assertEqual(runtime.state.turn_state, TurnState.ASSISTANT_SPEAKING)

        audio.push(frame(1, 6_000))
        await wait_until(lambda: runtime.state.turn_state is TurnState.USER_SPEAKING)
        self.assertTrue(playback.stopped)
        self.assertTrue(tts.cancelled)
        self.assertTrue(dialogue.cancelled)
        await runtime.close()

    async def test_multiple_speech_segments_keep_one_uncommitted_buffer(self) -> None:
        audio = FakeAudioInput()
        vad = FakeVAD(
            [
                VADEvent(VADEventKind.SPEECH_STARTED),
                VADEvent(VADEventKind.SPEECH_STOPPED),
                VADEvent(VADEventKind.SPEECH_STARTED),
                VADEvent(VADEventKind.SPEECH_STOPPED),
            ]
        )
        dialogue, tts = MockDialogueProvider(), MockTTSProvider()
        runtime = System1Runtime(audio_input=audio, vad=vad, dialogue=dialogue, tts=tts)
        await runtime.start()
        for sequence in range(1, 5):
            audio.push(frame(sequence, 5_000 if sequence in {1, 3} else 0))

        await wait_until(
            lambda: audio.queue_size == 0 and runtime.state.turn_state is TurnState.AWAITING_COMMIT
        )
        self.assertEqual(runtime.turn_audio_buffer.turn_id, "turn-1")
        self.assertEqual(runtime.turn_audio_buffer.frame_count, 4)
        self.assertEqual(dialogue.stream_calls, 0)
        self.assertEqual(tts.stream_calls, 0)
        await runtime.close()

    async def test_audio_queue_backpressure_is_bounded_and_reported(self) -> None:
        audio = FakeAudioInput(queue_size=2)
        runtime = System1Runtime(audio_input=audio, vad=FakeVAD([]))
        events = runtime.events.subscribe()
        await runtime.start()
        for sequence in range(20):
            audio.push(frame(sequence))

        self.assertLessEqual(audio.queue_size, 2)
        self.assertEqual(audio.frames_dropped, 18)
        await wait_until(lambda: audio.queue_size == 0)
        self.assertFalse(runtime._audio_task.done())
        snapshot = runtime.debug_snapshot()
        self.assertEqual(snapshot["frames_dropped"], 18)
        seen = []
        while not events.empty():
            seen.append((await events.get()).name)
        self.assertIn("audio.frame_dropped", seen)
        await runtime.close()

    async def test_runtime_can_attach_an_audio_input_after_start(self) -> None:
        runtime = System1Runtime()
        await runtime.start()
        audio = FakeAudioInput()
        vad = FakeVAD([VADEvent(VADEventKind.SPEECH_STARTED, 0.9)])
        await runtime.configure_audio_input(
            audio,
            vad,
            config=System1Config(audio=AudioConfig(device="test-device")),
        )
        audio.push(frame(1, 5_000))

        await wait_until(lambda: runtime.state.turn_state is TurnState.USER_SPEAKING)
        self.assertEqual(runtime.debug_snapshot()["audio_device"], "test-device")
        self.assertTrue(runtime.debug_snapshot()["audio_connected"])
        await runtime.close()
