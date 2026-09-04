"""An in-process async bus; the event contracts can outlive this transport."""

from __future__ import annotations

import asyncio

from .events import DomainEvent


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[DomainEvent]] = set()

    def subscribe(self) -> asyncio.Queue[DomainEvent]:
        queue: asyncio.Queue[DomainEvent] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[DomainEvent]) -> None:
        self._subscribers.discard(queue)

    async def publish(self, event: DomainEvent) -> None:
        for queue in self._subscribers.copy():
            await queue.put(event)
