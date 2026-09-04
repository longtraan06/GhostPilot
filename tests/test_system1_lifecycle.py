import asyncio
import unittest

from ghostpilot.system1.mock_providers import MockDialogueProvider, MockPlayback, MockTTSProvider
from ghostpilot.system1.providers import DialogueOutput
from ghostpilot.system1.runtime import System1Runtime, default_provider_registry
from ghostpilot.system1.config import System1Config
from ghostpilot.system1.state import AssistantState, TurnState, UserState


class System1LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_can_select_mock_adapters_from_configuration(self) -> None:
        runtime = System1Runtime.from_config(System1Config(), default_provider_registry())
        await runtime.start()
        self.assertTrue(runtime.stt.connected)
        await runtime.close()

    async def test_normal_turn_streams_to_speech_and_returns_to_listening(self) -> None:
        runtime = System1Runtime(
            dialogue=MockDialogueProvider([DialogueOutput("Sure, I found it.")]),
        )
        events = runtime.events.subscribe()
        await runtime.start()

        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped("find it")
        await runtime.wait_for_response()

        seen = []
        while not events.empty():
            seen.append((await events.get()).name)
        self.assertEqual(runtime.state.turn_state, TurnState.LISTENING)
        self.assertEqual(runtime.state.user_state, UserState.IDLE)
        self.assertEqual(runtime.state.assistant_state, AssistantState.IDLE)
        self.assertLess(seen.index("conversation.turn_committed"), seen.index("generation.started"))
        self.assertLess(seen.index("generation.started"), seen.index("speech.started"))
        self.assertLess(seen.index("speech.started"), seen.index("speech.finished"))
        await runtime.close()

    async def test_barge_in_stops_playback_then_cancels_providers_and_gives_user_turn(self) -> None:
        dialogue = MockDialogueProvider([DialogueOutput("I am still talking.")], delay=0.05)
        tts = MockTTSProvider(delay=0.2)
        playback = MockPlayback()
        runtime = System1Runtime(dialogue=dialogue, tts=tts, playback=playback)
        events = runtime.events.subscribe()
        await runtime.start()

        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped("say something")
        await asyncio.sleep(0.08)  # Dialogue yielded; streaming TTS is active.
        self.assertEqual(runtime.state.turn_state, TurnState.ASSISTANT_SPEAKING)

        await runtime.on_user_speech_started()

        self.assertTrue(playback.stopped)
        self.assertTrue(tts.cancelled)
        self.assertTrue(dialogue.cancelled)
        self.assertEqual(runtime.state.turn_state, TurnState.USER_SPEAKING)
        self.assertEqual(runtime.state.user_state, UserState.SPEAKING)
        seen = []
        while not events.empty():
            seen.append((await events.get()).name)
        self.assertIn("conversation.interrupted", seen)
        self.assertIn("generation.cancelled", seen)
        await runtime.close()
