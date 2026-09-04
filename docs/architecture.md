# GhostPilot Architecture

## Goal

GhostPilot is a realtime ambient multimodal personal agent.

Primary requirement:
low interaction latency.

Task completion latency is secondary.

## Main Components

### System 1
Realtime interaction runtime.

Responsibilities:
- audio
- VAD
- STT
- turn detection
- interruption
- dialogue
- TTS
- playback

### Orchestrator
Global coordinator.

Responsibilities:
- session state
- task registry
- routing
- communication between System 1 and other modules

### System 2
Future cognitive runtime.

Responsibilities:
- research
- coding
- reasoning
- planning
- deep vision tasks

### Tools
Future capability layer.

Examples:
- web
- browser
- terminal
- filesystem

### Vision
Future passive screen-perception module.

## Communication

System 1 communicates with the Orchestrator using typed events.

System 1 must not directly depend on System 2 implementation.

## Provider Strategy

STTProvider
DialogueProvider
TTSProvider

Providers must be replaceable through configuration.