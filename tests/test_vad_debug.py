import io
import time
import unittest
import wave

from ghostpilot.system1.audio import AudioFrame
from ghostpilot.system1.vad_debug import MicrophoneTestRecorder, PAGE


def audio_frame(data: bytes) -> AudioFrame:
    return AudioFrame(
        data=data,
        sample_rate=16_000,
        channels=1,
        timestamp=time.monotonic(),
    )


class MicrophoneTestRecorderTests(unittest.TestCase):
    def test_dashboard_exposes_reconnect_recovery_and_absolute_deadline(self) -> None:
        self.assertIn('id="recoveryBadge"', PAGE)
        self.assertIn("ACTIVE TURN INVALIDATED DUE TO STT RECONNECT", PAGE)
        self.assertIn('id="endpointDeadline"', PAGE)
        self.assertIn("endpoint_deadline_monotonic", PAGE)

    def test_canonical_frames_are_returned_as_playable_wav(self) -> None:
        recorder = MicrophoneTestRecorder()
        recorder.start(1)
        pcm = (2_000).to_bytes(2, "little", signed=True) * 320
        for _ in range(50):
            recorder.observe(audio_frame(pcm))

        wav_bytes, duration = recorder.finish_wav()
        with wave.open(io.BytesIO(wav_bytes), "rb") as recorded:
            self.assertEqual(recorded.getnchannels(), 1)
            self.assertEqual(recorded.getsampwidth(), 2)
            self.assertEqual(recorded.getframerate(), 16_000)
            self.assertEqual(recorded.getnframes(), 16_000)
        self.assertAlmostEqual(duration, 1.0)

    def test_capture_is_bounded_and_ignores_frames_when_inactive(self) -> None:
        recorder = MicrophoneTestRecorder()
        recorder.observe(audio_frame(b"\x00\x00" * 320))
        recorder.start(1)
        recorder.observe(audio_frame(b"\x01\x00" * 32_000))
        wav_bytes, duration = recorder.finish_wav()
        recorder.observe(audio_frame(b"\x02\x00" * 320))

        with wave.open(io.BytesIO(wav_bytes), "rb") as recorded:
            self.assertEqual(recorded.getnframes(), 16_000)
        self.assertEqual(duration, 1.0)

    def test_overlapping_recordings_are_rejected(self) -> None:
        recorder = MicrophoneTestRecorder()
        recorder.start(1)
        with self.assertRaises(RuntimeError):
            recorder.start(1)
