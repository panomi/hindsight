"""Return a playable URL for a video clip range.

The frontend's <video> element handles HTML5 byte-range requests against
/api/videos/:id/file natively, so we just return that URL with media
fragments. Time bounds are advisory.
"""
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import ToolResult
from app.models import Video

SCHEMA = {
    "name": "get_video_clip_url",
    "description": "Get a playable URL for a video clip between start and end "
                   "seconds. Use to surface evidence in the investigation UI.",
    "input_schema": {
        "type": "object",
        "properties": {
            "video_id": {"type": "string"},
            "start_sec": {"type": "number"},
            "end_sec": {"type": "number"},
        },
        "required": ["video_id", "start_sec", "end_sec"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    video_id = UUID(params["video_id"])
    start = float(params["start_sec"])
    end = float(params["end_sec"])

    v: Video | None = await session.get(Video, video_id)
    if v is None:
        return ToolResult(model_summary="get_video_clip_url: video not found",
                          ui_payload={}, error="not_found")

    url = f"/api/videos/{video_id}/file#t={start},{end}"
    return ToolResult(
        model_summary=f"get_video_clip_url({v.filename}, {start:.1f}-{end:.1f}s): {url}",
        ui_payload={"url": url, "video_id": str(video_id),
                    "start_sec": start, "end_sec": end,
                    "filename": v.filename},
    )
