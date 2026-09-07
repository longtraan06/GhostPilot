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
- conversation.turn_aborted
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

### Pre-roll protection

VAD requires several frames to decide that speech has started. While System 1
listens, `AudioPreRollBuffer` continuously retains the latest 250 ms of
canonical microphone frames. On `SPEECH_STARTED`, a newly created
`TurnAudioBuffer` receives that snapshot, including the trigger frame, before
live frames continue. This protects the first phonemes, such as `Hel` in
"Hello GhostPilot", from VAD-decision clipping.

During `AWAITING_COMMIT`, the same turn buffer remains active through short VAD
pauses and speech resumption. The pre-roll buffer is bounded, carries frame
references rather than PCM copies, and never reaches the EventBus. Milestone 3
can consume the complete `TurnAudioBuffer.snapshot()` after commit. The
PortAudio callback was not changed for this cleanup; native-format capture and
resampling remain its principal audio-path latency risk.

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

## Milestone 3A: mock realtime STT and endpointing

Audio frames fan out in parallel to VAD and `STTProvider`. Mock STT emits
snapshot-style `STTEvent` values: partial text produces `transcript.partial`,
and stable text produces `transcript.final`. `TranscriptManager` is scoped to a
turn and replaces overlapping snapshot updates rather than concatenating them.
Only a final from the latest speech segment is an endpoint commit candidate.

VAD stop is still only acoustic information. It enters `AWAITING_COMMIT` and
starts a cancellation-aware endpoint timer (600 ms by default). The M3A
`EndpointDetector` commits only if a final transcript is available when the
timer expires. A partial-only pause remains uncommitted. Speech resumption
cancels the timer and preserves the same turn ID, turn audio, and transcript.
Delayed STT events carry optional turn IDs and are ignored unless they match
the active transcript turn.

### M3A.1: turns and speech segments

A conversation turn is not the same thing as a VAD speech segment. A person
may pause briefly and continue the same turn, so resuming from
`AWAITING_COMMIT` retains its turn ID, audio buffer, and earlier transcript
snapshots while starting the next monotonic speech segment. Starting that
segment resets `latest_segment_final`; an older final is still useful history,
but cannot authorize endpoint commit.

Endpointing therefore requires a valid final for the latest segment, rather
than merely any final in the turn. A provider-neutral optional `segment_id` on
an STT event lets an adapter associate delayed network results with the speech
segment that produced them. Runtime rejects a result whose turn or segment is
no longer active. This preserves the mock contract today and gives future
provider adapters a safe normalization point without embedding vendor logic in
the runtime.

For manual mock testing, start the existing VAD debug UI, select an input, and
speak until a user turn starts. Enter text in **Inject mock transcript**, send
one or more partials and then a final. Stop speaking: the dashboard shows
`AWAITING_COMMIT` / `WAITING`, then `COMMITTED` after the endpoint timeout.
The equivalent development endpoint is `POST /api/mock-transcript` with JSON
`{"text":"hello ghostpilot","is_final":true}`; it is only available when
the active provider is `MockSTTProvider`.

## Milestone 3B: self-hosted Nemotron streaming STT

`NemotronSTTProvider` is the vendor/protocol adapter for the self-hosted
Nemotron service. It owns health validation, one persistent WebSocket,
binary/control framing, a bounded audio send queue, response parsing, service
diagnostics, and reconnect backoff. System 1, VAD, transcript management,
endpointing, and turn ownership do not import WebSocket details.

At startup the adapter validates `/health`: `ok` and `warmed_up` must be true,
the sample rate must be 16 kHz, lookahead must be 1, and protocol version 2
must advertise `explicit_segment_id`. It then connects to the stream endpoint,
waits briefly for `ready`, and keeps reconnecting in the background if the
service is unavailable. Each successful connection has a monotonic generation;
receive work from older connections is rejected and the new server-side decoder
is reset. Provider errors become typed System 1 status and failure events rather
than crashing the runtime.

Microphone frames remain canonical PCM16 little-endian, mono, 16 kHz. VAD and
STT consume the same asyncio audio path, outside the PortAudio callback. STT is
speech-gated: when VAD starts a segment, the bounded pre-roll is queued first,
then live frames are queued while speech remains active. Silence outside a
speech segment is not sent. Queue overflow drops frames deterministically and
is visible in diagnostics instead of blocking capture or growing memory.

At VAD speech start, the adapter first sends
`{"type":"segment_start","segment_id":LOCAL_ID}` before pre-roll or PCM.
The service must echo this exact ID in partial/final messages; its private
decoder generation is never exposed. VAD `SPEECH_STOPPED` then sends the
provider-neutral `end_segment()` control, mapped to `{"type":"segment_end"}`.
A segment final is not a user-turn commit. Only EndpointDetector may commit
after its timeout and latest-segment-final checks. Local turn state is committed
once, then `commit_turn()` sends the server control; `turn_final` is treated
only as an acknowledgement and can never trigger a second commit.

Configure and run real STT mode in PowerShell:

```powershell
$env:GHOSTPILOT_STT_PROVIDER = "nemotron"
$env:GHOSTPILOT_STT_WS_URL = "ws://localhost:6010/v1/stt/stream"
$env:GHOSTPILOT_STT_HEALTH_URL = "http://localhost:6010/health"
python -m ghostpilot.system1.vad_debug --port 8765
```

Open `http://127.0.0.1:8765`, select an input device, and use the M3B dashboard
to inspect System 1/VAD state, service health and reconnects, partial/final/best
transcripts, turn and segment finality, endpoint timer, audio queue/backpressure,
latencies, and the bounded event timeline. Reconnect, reset, and ping controls
operate only on the STT adapter and do not bypass System 1 turn ownership.

The **Microphone + Listen Back** panel is an explicit local diagnostic. After
an input is selected, **Record 4 seconds** taps the canonical frames in the
asyncio runtime path, keeps at most the requested bounded duration, and returns
a PCM16 mono WAV for playback in the browser. It does not use browser microphone
capture, does not place PCM on the EventBus, and does not add work to the
PortAudio callback. This separates wrong/quiet input-device problems from STT
service or protocol problems.

For protocol diagnosis, run the dashboard with `--debug-stt`. Console logging
records the first PCM send and one-second aggregate progress, each control after
it is actually written to the WebSocket, every response type/payload, and any
turn/segment/generation rejection. It still does not log every 20 ms frame.
The web **STT Protocol Debug** and **STT Protocol Trace** panels expose the same
essential counters: frames queued/sent for the active segment, last control,
last response type, partial/final totals, ignored responses, and the exact last
rejection reason. `PCM SENT · NO TRANSCRIPT RESPONSE` means capture and client
WebSocket sending succeeded but no partial/final was received for that segment.

Resumed speech uses frame ownership rather than replaying the full pre-roll.
`AudioFrame.sequence` is the primary boundary: after a speech stop, only frames
whose sequence is newer than that stop are sent as the next segment's pre-roll.
The monotonic timestamp is the fallback for synthetic inputs without positive
sequences. A new user turn still receives the full configured pre-roll.

Endpoint deadlines are absolute and belong to one speech-stop boundary. If the
timer expires before a final transcript arrives, the deadline remains recorded;
a valid final for that same turn and segment commits immediately instead of
starting another 600 ms timer. Speech resumption cancels and invalidates the old
deadline before the next segment creates a new one.

A WebSocket disconnect invalidates an active `USER_SPEAKING` or
`AWAITING_COMMIT` turn. System 1 cancels endpointing, emits
`conversation.turn_aborted`, clears transcript/audio ownership, resets the STT
adapter, and returns to `LISTENING`. No transcript is committed and no dialogue
generation starts. Responses on the replacement connection are rejected until
a fresh user turn binds a new segment.

The outbound adapter uses one bounded, typed FIFO. Audio keeps the configured
queue limit and may be dropped under backpressure; a small bounded reserve is
kept for controls. When the total queue is full, a control may evict only the
oldest audio item, never another control. FIFO placement provides the segment
barrier: accepted audio before `segment_end` is sent before it, and future
segment audio is sent after it.

The debug dashboard displays the absolute monotonic endpoint deadline and
remaining time, connection generation, active STT binding, drop counters, and a
prominent `ACTIVE TURN INVALIDATED DUE TO STT RECONNECT` recovery indicator.

The adapter currently assumes the service's transcript text follows the
existing full-turn snapshot contract. A service that returns segment-local text
must normalize/merge it in its adapter rather than adding provider-specific
merging to `TranscriptManager`.

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
