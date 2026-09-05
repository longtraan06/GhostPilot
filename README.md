# GhostPilot

GhostPilot is a latency-first ambient personal agent. This repository contains
the vendor-neutral, asyncio-based System 1 interaction runtime through M3B,
including real microphone/VAD and self-hosted streaming STT support.

## Quick start

```powershell
python -m unittest discover -s tests -v
```

Install the local microphone/VAD dashboard with `pip install -r requirements.txt`.
`requirements.lock` records the exact Windows/Python 3.12 package set verified
for this repository.

## Layout

```text
src/ghostpilot/system1/
  adapters/           sounddevice and Nemotron protocol isolation boundary
  audio.py            canonical frames and bounded per-turn buffer
  config.py           provider selection registry
  event_bus.py        in-process async event transport
  events.py           typed System 1 event contracts
  interruption.py     local-first barge-in path
  mock_providers.py   deterministic development adapters
  providers.py        STT, dialogue, TTS, and playback contracts
  runtime.py          composition root
  speech.py           streaming text segmentation
  state.py            conversation state machine
  turn.py             turn and response orchestration
  vad.py              vendor-neutral local VAD contract and baseline
  vad_debug.py        development-only FastAPI/WebSocket dashboard
tests/test_system1_lifecycle.py
```

## Next adapters

Implement a real adapter only inside the provider layer: an STT adapter fulfils
`STTProvider`, a Gemini Flash adapter fulfils `DialogueProvider`, and a chosen
streaming voice adapter fulfils `TTSProvider`. Register the adapter factory in
`ProviderRegistry`, select its name in `System1Config`, compose with
`System1Runtime.from_config`, and keep the turn,
interruption, and runtime modules unchanged.

System 1 emits proposed dialogue actions but does not execute them. The future
Orchestrator may subscribe to those events and route them to System 2 or tools.

VAD emits `audio.speech_stopped` without committing a turn. A future endpoint
detector must explicitly call `System1Runtime.commit_turn(transcript)` before
generation begins.
