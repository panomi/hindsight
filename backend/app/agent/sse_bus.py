"""Per-investigation SSE event bus.

The orchestrator writes events into a queue; the SSE endpoint reads from
the same queue and streams to the frontend. We keep the queue in-process
because investigations are pinned to a single API process for simplicity.
"""
import asyncio
from collections import defaultdict
from uuid import UUID

# investigation_id -> queue
_QUEUES: dict[UUID, asyncio.Queue] = defaultdict(lambda: asyncio.Queue(maxsize=256))


def queue_for(investigation_id: UUID) -> asyncio.Queue:
    return _QUEUES[investigation_id]


def reset(investigation_id: UUID) -> None:
    _QUEUES.pop(investigation_id, None)
