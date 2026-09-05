"""An in-process async bus; the event contracts can outlive this transport."""

from __future__ import annotations

import asyncio

from .events import DomainEvent


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[DomainEvent]] = set()

    def subscribe(self, *, maxsize: int = 0) -> asyncio.Queue[DomainEvent]:
        queue: asyncio.Queue[DomainEvent] = asyncio.Queue(maxsize)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[DomainEvent]) -> None:
        self._subscribers.discard(queue)

    async def publish(self, event: DomainEvent) -> None:
        for queue in self._subscribers.copy():
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)
