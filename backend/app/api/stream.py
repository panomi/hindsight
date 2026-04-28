"""SSE endpoint that streams agent events to the frontend."""
import asyncio
import json
from uuid import UUID

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.agent import sse_bus

router = APIRouter(prefix="/api/investigations", tags=["stream"])


@router.get("/{investigation_id}/stream")
async def stream(investigation_id: UUID):
    async def gen():
        q = sse_bus.queue_for(investigation_id)
        while True:
            try:
                evt = await asyncio.wait_for(q.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # heartbeat keeps proxies happy
                yield {"event": "ping", "data": "{}"}
                continue
            yield {"event": evt["event"], "data": json.dumps(evt["data"])}
            # Do NOT break on "done" — keep the connection alive so the
            # frontend can receive events from follow-up messages without
            # needing to reconnect.
    return EventSourceResponse(gen())
