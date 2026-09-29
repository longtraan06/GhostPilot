import unittest

from ghostpilot.system1.config import DialogueContextConfig
from ghostpilot.system1.dialogue_context import ConversationHistory, DialogueContextBuilder


def build(history: ConversationHistory, current: str, *, exchanges=4, chars=6_000):
    return DialogueContextBuilder(
        DialogueContextConfig(
            system_prompt="system prompt",
            max_history_exchanges=exchanges,
            max_history_chars=chars,
        )
    ).build(history, current)


class DialogueContextTests(unittest.TestCase):
    def test_empty_history_always_contains_system_and_current_user(self):
        context = build(ConversationHistory(), "Hello")
        self.assertEqual(
            [(message.role, message.content) for message in context.messages],
            [("system", "system prompt"), ("user", "Hello")],
        )
        self.assertEqual(context.historical_exchanges, 0)

    def test_completed_exchange_is_emitted_chronologically_before_current_user(self):
        history = ConversationHistory()
        history.append_exchange("My name is Alex.", "Nice to meet you.")
        context = build(history, "What name did I tell you?")
        self.assertEqual(
            [(message.role, message.content) for message in context.messages],
            [
                ("system", "system prompt"),
                ("user", "My name is Alex."),
                ("assistant", "Nice to meet you."),
                ("user", "What name did I tell you?"),
            ],
        )

    def test_exchange_count_budget_keeps_newest_complete_exchanges(self):
        history = ConversationHistory()
        for index in range(1, 6):
            history.append_exchange(f"u{index}", f"a{index}")
        context = build(history, "current", exchanges=4)
        self.assertEqual(
            [(message.role, message.content) for message in context.messages],
            [
                ("system", "system prompt"),
                ("user", "u2"), ("assistant", "a2"),
                ("user", "u3"), ("assistant", "a3"),
                ("user", "u4"), ("assistant", "a4"),
                ("user", "u5"), ("assistant", "a5"),
                ("user", "current"),
            ],
        )

    def test_character_budget_drops_whole_old_exchanges_without_orphans(self):
        history = ConversationHistory()
        history.append_exchange("old-user", "old-assistant")
        history.append_exchange("new-user", "new-assistant")
        context = build(history, "current", chars=len("new-usernew-assistant"))
        history_messages = context.messages[1:-1]
        self.assertEqual(
            [(message.role, message.content) for message in history_messages],
            [("user", "new-user"), ("assistant", "new-assistant")],
        )
        self.assertEqual(context.historical_chars, len("new-usernew-assistant"))

    def test_large_single_exchange_is_skipped_and_current_user_still_remains(self):
        history = ConversationHistory()
        history.append_exchange("x" * 20, "y" * 20)
        context = build(history, "current user is always retained", chars=10)
        self.assertEqual(context.historical_exchanges, 0)
        self.assertEqual(context.messages[0].role, "system")
        self.assertEqual(context.messages[-1].content, "current user is always retained")

    def test_oversized_newest_exchange_does_not_hide_an_older_eligible_exchange(self):
        history = ConversationHistory()
        history.append_exchange("old", "fits")
        history.append_exchange("x" * 20, "y" * 20)
        context = build(history, "current", chars=len("oldfits"))
        self.assertEqual(
            [(message.role, message.content) for message in context.messages],
            [
                ("system", "system prompt"),
                ("user", "old"),
                ("assistant", "fits"),
                ("user", "current"),
            ],
        )

    def test_clear_resets_counts_and_context_metrics(self):
        history = ConversationHistory()
        history.append_exchange("u", "a")
        history.clear()
        context = build(history, "current")
        self.assertEqual(history.exchange_count, 0)
        self.assertEqual(history.message_count, 0)
        self.assertEqual(history.character_count, 0)
        self.assertEqual(context.total_messages, 2)
        self.assertEqual(context.estimated_tokens, 5)

    def test_history_storage_is_bounded_by_its_context_policy(self):
        config = DialogueContextConfig(
            system_prompt="system prompt", max_history_exchanges=2, max_history_chars=21
        )
        history = ConversationHistory(config)
        history.append_exchange("old", "answer")
        history.append_exchange("middle", "answer")
        history.append_exchange("new", "answer")

        self.assertEqual(
            [(exchange.user, exchange.assistant) for exchange in history.exchanges()],
            [("middle", "answer"), ("new", "answer")],
        )
