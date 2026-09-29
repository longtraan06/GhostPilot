"""Measure real streaming dialogue latency for several bounded context sizes.

Run after configuring ``.env`` with a reachable OpenAI-compatible server:

    .\\.venv\\Scripts\\python.exe scripts\\benchmark_dialogue_context.py
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from ghostpilot.system1.adapters.openai_compatible_dialogue import (
    OpenAICompatibleDialogueProvider,
)
from ghostpilot.system1.config import System1Config
from ghostpilot.system1.dialogue_context import ConversationHistory, DialogueContextBuilder
from ghostpilot.system1.providers import DialogueProviderError


EXCHANGES = (
    ("My preferred programming language is Python.", "Noted: Python is your preference."),
    ("Keep answers brief unless I ask for detail.", "I will keep them concise by default."),
    ("I am testing a realtime voice assistant.", "I will prioritize a direct, spoken response."),
    ("The assistant should remember this only in the current session.", "Yes, this context is in memory only."),
)


async def measure(max_exchanges: int) -> None:
    system_config = System1Config.from_env()
    if not system_config.dialogue.model:
        raise SystemExit("Set GHOSTPILOT_DIALOGUE_MODEL in .env before benchmarking.")

    context_config = replace(
        system_config.dialogue_context, max_history_exchanges=max_exchanges
    )
    history = ConversationHistory()
    for user, assistant in EXCHANGES:
        history.append_exchange(user, assistant)
    context = DialogueContextBuilder(context_config).build(
        history, "What programming language do I prefer? Answer in one short sentence."
    )
    provider = OpenAICompatibleDialogueProvider(system_config.dialogue)
    text = "".join([output.text async for output in provider.stream(context.messages)])
    diagnostics = provider.diagnostics()
    print(
        f"history limit={max_exchanges}: "
        f"messages={context.total_messages}, chars={context.total_chars}, "
        f"estimated_tokens={context.estimated_tokens}, "
        f"TTFT={diagnostics['last_ttft_ms']} ms, "
        f"total={diagnostics['last_total_duration_ms']} ms\n"
        f"  reply: {text}"
    )


async def main() -> None:
    try:
        for max_exchanges in (0, 1, 2, 4):
            await measure(max_exchanges)
    except DialogueProviderError as error:
        raise SystemExit(
            "Dialogue benchmark could not reach the configured server. "
            "Check GHOSTPILOT_DIALOGUE_BASE_URL and start the model first: "
            f"{error}"
        ) from error


if __name__ == "__main__":
    asyncio.run(main())
