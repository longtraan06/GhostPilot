# GhostPilot Development Rules

- System 1 owns realtime interaction.
- Orchestrator owns global task/application state.
- System 2 handles deep reasoning and long-running cognitive tasks.
- STT, Dialogue, and TTS providers must be replaceable.
- Vendor-specific SDK code must stay inside provider adapters.
- Optimize for latency first.
- Language is not a constraint; English-first is acceptable.
- Barge-in must stop playback immediately before waiting for cloud cancellation.
- Use async, cancellation-aware code.
- Start as a modular monolith.
- Do not introduce microservices, Redis, NATS, Kafka, or Kubernetes yet.
- Do not implement System 2 until System 1 is stable.
- Run tests after architectural changes.