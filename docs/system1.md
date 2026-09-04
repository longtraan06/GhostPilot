# System 1 Design

## Goal

System 1 is the realtime interaction runtime of GhostPilot.

Priority:
minimum perceived latency.

## Pipeline

Mic
→ VAD
→ STT
→ Turn Manager
→ Dialogue Provider
→ Speech Segmenter
→ TTS
→ Playback

## Components

### VAD
Detect user speech start/stop.

### STTProvider

Interface for realtime speech recognition.

Possible providers:
- Deepgram
- ElevenLabs
- local model

No provider is fixed yet.

### TurnManager

Controls conversation ownership.

States:

- LISTENING
- USER_SPEAKING
- THINKING
- ASSISTANT_SPEAKING
- INTERRUPTED

### DialogueProvider

Fast conversational model.

Initial real implementation may use Gemini Flash.

Must support streaming and cancellation.

### SpeechSegmenter

Converts streamed LLM tokens into reasonable chunks for TTS.

Do not send individual tokens directly to TTS.

### TTSProvider

Streaming text-to-speech interface.

Provider not fixed yet.

### InterruptionController

When:

ASSISTANT_SPEAKING
+
USER_SPEECH_STARTED

Immediately:

1. stop playback
2. cancel TTS
3. cancel dialogue generation
4. transition to USER_SPEAKING

## Events

Initial event types:

- audio.speech_started
- audio.speech_stopped
- conversation.turn_started
- conversation.turn_committed
- conversation.interrupted
- generation.started
- generation.cancelled
- speech.started
- speech.finished

## Provider Interfaces

STTProvider:
- start()
- send_audio()
- events()
- close()

DialogueProvider:
- stream()
- cancel()

TTSProvider:
- stream()
- cancel()

## Milestone 1

Implement only:

- events
- conversation state
- state machine
- event bus
- provider interfaces
- mock STT
- mock Dialogue
- mock TTS
- TurnManager
- InterruptionController
- System1Runtime
- tests

No real STT or TTS API yet.

## Milestone 2: local microphone and VAD

### Canonical audio format

System 1 accepts microphone audio only as `AudioFrame` objects in this format:

- 16,000 Hz sample rate
- one channel (mono)
- signed PCM16 / `int16`
- 20 ms frames by default (configurable to a nearby fixed duration)

`AudioInput` is a vendor-neutral asynchronous source with `start()`, `frames()`,
and `close()`. The optional `SoundDeviceAudioInput` adapter owns all
`sounddevice`/PortAudio code and normalizes captured input to that format.
It captures at the selected device's native sample rate, downmixes to mono, and
resamples each fixed frame to 16 kHz before it reaches the realtime runtime.
Frames travel directly through the realtime audio loop; they never pass through
the general `EventBus`.

### VAD and endpointing

`VoiceActivityDetector` only produces acoustic `SPEECH_STARTED` and
`SPEECH_STOPPED` signals. The default local implementation is a lightweight
energy detector with configurable speech threshold, minimum speech duration,
and minimum silence duration. It can be replaced by a Silero adapter without
changing runtime or turn-management code.

VAD stop is not endpoint detection. A speech stop moves the conversation to
`AWAITING_COMMIT`, emits `audio.speech_stopped`, and retains its audio. Only a
future endpoint detector may call `System1Runtime.commit_turn(transcript)`.

### Buffer lifecycle

`TurnAudioBuffer` keeps bounded PCM frames for the active, uncommitted user
turn. Short VAD pauses resume the same buffer; a new user turn creates a new
buffer. Milestone 3's STT adapter will consume the buffer snapshot or stream
from this same ownership boundary. When its duration limit is exceeded, the
oldest frames are dropped deterministically and counted.

### VAD debug UI

The development-only FastAPI/WebSocket UI displays backend-produced VAD state,
audio level, probability, System 1 turn state, semantic events, and small
buffer/queue metrics. It does not run browser VAD or send microphone PCM to the
browser.

Install the optional local dependencies, then run:

```bash
pip install -e ".[audio,vad-debug]"
python -m ghostpilot.system1.vad_debug
```

Open `http://127.0.0.1:8000`. Use `--device`, `--threshold`, or `--port` when
needed. Run `python -m ghostpilot.system1.vad_debug --list-devices` to show
available microphones, then pass its numeric ID with `--device ID`. Remain
silent to observe `LISTENING`; speak to observe `SPEAKING`; stop to observe
endpoint-waiting behavior without dialogue generation.

The dashboard also has an **Input device** dropdown. It lists only devices with
input channels; choosing one and pressing **Use selected input** swaps the
running `AudioInput` adapter while keeping the System 1 runtime alive.

## Required Tests

Normal flow:

USER_SPEECH_STARTED
→ USER_SPEECH_STOPPED
→ TURN_COMMITTED
→ THINKING
→ ASSISTANT_SPEAKING
→ LISTENING

Barge-in:

ASSISTANT_SPEAKING
→ USER_SPEECH_STARTED
→ playback cancelled
→ TTS cancelled
→ dialogue generation cancelled
→ USER_SPEAKING
