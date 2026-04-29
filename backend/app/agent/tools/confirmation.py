"""Pause the agent and ask the user to confirm/reject a candidate set.

The orchestrator emits a `confirmation_request` SSE event and awaits an
asyncio.Event keyed on a confirmation_id. The user's POST to
/api/investigations/:id/confirm sets that event and stores the selection
in a small in-memory pending-store, which this tool reads back.
"""
import asyncio
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import ToolResult

SCHEMA = {
    "name": "request_user_confirmation",
    "description": (
        "Ask the user to confirm/reject candidate items. Use specific prompts: "
        "not 'are these the right people?' but e.g. 'These are 8 person tracks "
        "that match the reference image — confirm matches and reject false "
        "positives'. mode='instances' when proposing tracks for subject "
        "registration; 'frames' for ambiguous retrieval; 'events' for temporal "
        "grouping. "
        "IMPORTANT — to enable thumbnails in the UI, every item MUST carry "
        "either 'frame_id' (preferred) OR both 'video_id' and "
        "'timestamp_seconds'. "
        "For mode='instances' (subject registration): the 'id' field MUST be "
        "the Detection UUID (set 'id' = detection_id explicitly so the user's "
        "selection is what register_subject will accept). Frame IDs and shot "
        "IDs will NOT work for subject registration — only person-crop "
        "detections have the box_embeddings register_subject needs. Also "
        "include 'instance_id' so the UI can show the track number badge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": (
                    "List of items the user will confirm/reject. Each item "
                    "should be a dict with at minimum a stable 'id' (or "
                    "'detection_id' / 'frame_id') plus visual context "
                    "('frame_id' OR 'video_id'+'timestamp_seconds') so the "
                    "UI can render a thumbnail."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "detection_id": {"type": "string"},
                        "frame_id": {"type": "string"},
                        "video_id": {"type": "string"},
                        "timestamp_seconds": {"type": "number"},
                        "instance_id": {"type": "integer"},
                        "label": {"type": "string"},
                    },
                },
            },
            "question": {"type": "string"},
            "mode": {"type": "string", "enum": ["frames", "instances", "events"],
                     "default": "frames"},
        },
        "required": ["items", "question"],
    },
}


# In-process registry — a real deployment would use Redis to survive restarts.
# Map confirmation_id -> (asyncio.Event, response_dict)
PENDING: dict[UUID, tuple[asyncio.Event, dict]] = {}


def register_pending(confirmation_id: UUID, event: asyncio.Event) -> None:
    PENDING[confirmation_id] = (event, {})


def resolve_pending(confirmation_id: UUID, payload: dict) -> bool:
    entry = PENDING.get(confirmation_id)
    if entry is None:
        return False
    event, store = entry
    store.update(payload)
    event.set()
    return True


def get_resolution(confirmation_id: UUID) -> dict | None:
    entry = PENDING.get(confirmation_id)
    return entry[1] if entry else None


async def run(session: AsyncSession, params: dict, investigation_id: UUID,
              sse_queue: asyncio.Queue | None = None,
              timeout_sec: float = 600.0) -> ToolResult:
    items = params["items"]
    question = params["question"]
    mode = params.get("mode", "frames")
    confirmation_id = uuid4()

    event = asyncio.Event()
    register_pending(confirmation_id, event)

    if sse_queue is not None:
        await sse_queue.put({
            "event": "confirmation_request",
            "data": {
                "confirmation_id": str(confirmation_id),
                "mode": mode, "question": question, "items": items,
            },
        })

    try:
        await asyncio.wait_for(event.wait(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        # Tell the UI the popup is dead so it stops showing the prompt
        # forever even though the agent has already moved on.
        if sse_queue is not None:
            await sse_queue.put({
                "event": "confirmation_resolved",
                "data": {
                    "confirmation_id": str(confirmation_id),
                    "reason": "timeout",
                },
            })
        return ToolResult(
            model_summary=f"request_user_confirmation: timed out after {timeout_sec}s",
            ui_payload={"confirmation_id": str(confirmation_id), "timed_out": True},
        )

    resolution = get_resolution(confirmation_id) or {}
    confirmed = resolution.get("confirmed_ids", [])
    rejected = resolution.get("rejected_ids", [])
    skipped = bool(resolution.get("skipped"))

    # Tell the UI this confirmation is no longer pending — covers cases where
    # multiple tabs have it open, or the user resolved via "skip" (sending a
    # new message while the popup was open).  The submitting tab also clears
    # locally on success; this is belt-and-braces for the rest.
    if sse_queue is not None:
        await sse_queue.put({
            "event": "confirmation_resolved",
            "data": {
                "confirmation_id": str(confirmation_id),
                "reason": "skipped" if skipped else "submitted",
            },
        })

    summary = (
        f"user skipped confirmation [{mode}] (moved on to a new question)"
        if skipped
        else f"user confirmation [{mode}]: {len(confirmed)} confirmed, {len(rejected)} rejected"
    )

    return ToolResult(
        model_summary=summary,
        ui_payload={
            "confirmation_id": str(confirmation_id),
            "confirmed_ids": confirmed, "rejected_ids": rejected,
            "mode": mode, "skipped": skipped,
        },
        top_k_used=len(confirmed) + len(rejected),
    )
