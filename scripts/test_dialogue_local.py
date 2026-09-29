"""Manual vLLM/OpenAI-compatible dialogue smoke test; excluded from unit tests."""

from __future__ import annotations

import asyncio
from ghostpilot.system1.adapters.openai_compatible_dialogue import (
    OpenAICompatibleDialogueProvider,
)
from ghostpilot.system1.config import System1Config
from ghostpilot.system1.dialogue_context import ConversationHistory, DialogueContextBuilder


async def main() -> None:
    system_config = System1Config.from_env()
    config = system_config.dialogue
    model = config.model
    if not model:
        raise SystemExit(
            "Set GHOSTPILOT_DIALOGUE_MODEL to an identifier returned by "
            "http://localhost:3308/v1/models."
        )
    provider = OpenAICompatibleDialogueProvider(config)
    context_builder = DialogueContextBuilder(system_config.dialogue_context)
    context = context_builder.build(
        ConversationHistory(), "In one short sentence, say hello."
    )
    print(f"model: {model}\nbase URL: {provider.diagnostics()['base_url']}")
    print("text: ", end="", flush=True)
    async for output in provider.stream(context.messages):
        print(output.text, end="", flush=True)
    diagnostics = provider.diagnostics()
    print(
        f"\nTTFT: {diagnostics['last_ttft_ms']} ms"
        f"\ntotal: {diagnostics['last_total_duration_ms']} ms"
    )

    pending = asyncio.create_task(
        _drain(
            provider,
            context_builder,
            "Give a deliberately long explanation of realtime systems.",
        )
    )
    await asyncio.sleep(0.1)
    await provider.cancel()
    await pending
    print(f"cancellation close latency: {provider.diagnostics()['last_cancellation_latency_ms']} ms")


async def _drain(
    provider: OpenAICompatibleDialogueProvider,
    context_builder: DialogueContextBuilder,
    prompt: str,
) -> None:
    context = context_builder.build(ConversationHistory(), prompt)
    async for _ in provider.stream(context.messages):
        pass


if __name__ == "__main__":
    asyncio.run(main())
