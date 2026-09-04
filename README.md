# GhostPilot

GhostPilot is a latency-first ambient personal agent. This repository currently
contains Milestone 1: the vendor-neutral, asyncio-based System 1 interaction
foundation.

## Quick start

```powershell
python -m unittest discover -s tests -v
```

## Layout

```text
src/ghostpilot/system1/
  adapters/           future vendor-SDK isolation boundary
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
