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