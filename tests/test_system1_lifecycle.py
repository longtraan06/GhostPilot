import asyncio
import unittest

from ghostpilot.system1.mock_providers import MockDialogueProvider, MockPlayback, MockTTSProvider
from ghostpilot.system1.providers import DialogueOutput
from ghostpilot.system1.runtime import System1Runtime, default_provider_registry
from ghostpilot.system1.adapters.nemotron_stt import NemotronSTTProvider
from ghostpilot.system1.config import System1Config
from ghostpilot.system1.state import AssistantState, TurnState, UserState


class System1LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_can_select_mock_adapters_from_configuration(self) -> None:
        runtime = System1Runtime.from_config(System1Config(), default_provider_registry())
        await runtime.start()
        self.assertTrue(runtime.stt.connected)
        await runtime.close()

    async def test_registry_can_select_nemotron_without_hard_wiring_runtime(self) -> None:
        config = System1Config(stt_provider="nemotron")
        provider = default_provider_registry(config).build(config.stt_provider)
        self.assertIsInstance(provider, NemotronSTTProvider)

    async def test_normal_turn_streams_to_speech_and_returns_to_listening(self) -> None:
        runtime = System1Runtime(
            dialogue=MockDialogueProvider([DialogueOutput("Sure, I found it.")]),
        )
        events = runtime.events.subscribe()
        await runtime.start()

        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        self.assertEqual(runtime.state.turn_state, TurnState.AWAITING_COMMIT)
        self.assertTrue(events.empty() is False)
        seen_before_commit = []
        while not events.empty():
            seen_before_commit.append((await events.get()).name)
        self.assertIn("audio.speech_stopped", seen_before_commit)
        self.assertNotIn("conversation.turn_committed", seen_before_commit)
        self.assertNotIn("generation.started", seen_before_commit)

        await runtime.commit_turn("find it")
        await runtime.wait_for_response()

        seen = seen_before_commit
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
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("say something")
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

    async def test_playback_recovers_for_response_after_barge_in(self) -> None:
        dialogue = MockDialogueProvider([DialogueOutput("A fresh response.")], delay=0.05)
        tts = MockTTSProvider(delay=0.2)
        playback = MockPlayback()
        runtime = System1Runtime(dialogue=dialogue, tts=tts, playback=playback)
        await runtime.start()

        await runtime.on_user_speech_started()
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("first request")
        await asyncio.sleep(0.08)
        self.assertEqual(runtime.state.turn_state, TurnState.ASSISTANT_SPEAKING)

        await runtime.on_user_speech_started()
        self.assertTrue(playback.stopped)
        await runtime.on_user_speech_stopped()
        await runtime.commit_turn("replacement request")
        await runtime.wait_for_response()

        self.assertTrue(playback.played)
        self.assertEqual(playback.played[-1].data, b"A fresh response.")
        self.assertFalse(playback.stopped)
        self.assertEqual(runtime.state.turn_state, TurnState.LISTENING)
        await runtime.close()
