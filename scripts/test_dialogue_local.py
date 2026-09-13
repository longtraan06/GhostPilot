"""Manual vLLM/OpenAI-compatible dialogue smoke test; excluded from unit tests."""

from __future__ import annotations

import asyncio
from ghostpilot.system1.adapters.openai_compatible_dialogue import (
    OpenAICompatibleDialogueProvider,
)
from ghostpilot.system1.config import System1Config


async def main() -> None:
    config = System1Config.from_env().dialogue
    model = config.model
    if not model:
        raise SystemExit(
            "Set GHOSTPILOT_DIALOGUE_MODEL to an identifier returned by "
            "http://localhost:3308/v1/models."
        )
    provider = OpenAICompatibleDialogueProvider(config)
    print(f"model: {model}\nbase URL: {provider.diagnostics()['base_url']}")
    print("text: ", end="", flush=True)
    async for output in provider.stream("In one short sentence, say hello."):
        print(output.text, end="", flush=True)
    diagnostics = provider.diagnostics()
    print(
        f"\nTTFT: {diagnostics['last_ttft_ms']} ms"
        f"\ntotal: {diagnostics['last_total_duration_ms']} ms"
    )

    pending = asyncio.create_task(
        _drain(provider, "Give a deliberately long explanation of realtime systems.")
    )
    await asyncio.sleep(0.1)
    await provider.cancel()
    await pending
    print(f"cancellation close latency: {provider.diagnostics()['last_cancellation_latency_ms']} ms")


async def _drain(provider: OpenAICompatibleDialogueProvider, prompt: str) -> None:
    async for _ in provider.stream(prompt):
        pass


if __name__ == "__main__":
    asyncio.run(main())
