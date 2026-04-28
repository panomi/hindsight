"""Query the pre-computed object detections by class/confidence/video/time.

Backed by an RT-DETRv2 inference pass at ingest time, restricted to the
COCO-80 closed class set.  For open-vocabulary concepts, use
`open_vocab_detect` instead.
"""
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import ToolResult
from app.agent.tools._utils import empty_scope_result, scope_videos
from app.config import get_settings
from app.models import Detection, Frame

settings = get_settings()

SCHEMA = {
    "name": "get_object_detections",
    "description": "Filter pre-computed object detections (COCO-80 closed class "
                   "set) by class, confidence, video, and time range. Use for "
                   "fixed categories (person, car, dog, …). For free-form "
                   "concepts use `open_vocab_detect` instead.",
    "input_schema": {
        "type": "object",
        "properties": {
            "video_ids": {"type": "array", "items": {"type": "string"}},
            "classes": {"type": "array", "items": {"type": "string"}},
            "min_confidence": {"type": "number", "default": 0.5},
            "time_range": {
                "type": "array", "items": {"type": "number"},
                "minItems": 2, "maxItems": 2,
                "description": "[start_sec, end_sec] within each video",
            },
            "top_k": {"type": "integer", "default": 50, "maximum": 50},
        },
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    video_ids = await scope_videos(session, investigation_id, params.get("video_ids", []))
    if not video_ids:
        return empty_scope_result("get_object_detections")
    classes = params.get("classes") or None
    min_conf = float(params.get("min_confidence", 0.5))
    time_range = params.get("time_range")
    top_k = min(int(params.get("top_k", 50)), settings.tool_result_top_k_default)

    conds = [
        Detection.confidence >= min_conf,
        Detection.video_id.in_(video_ids),
    ]
    if classes:
        conds.append(Detection.class_name.in_(classes))

    stmt = select(Detection).where(and_(*conds))
    if time_range:
        stmt = stmt.join(Frame, Frame.id == Detection.frame_id).where(
            and_(Frame.timestamp_seconds >= time_range[0],
                 Frame.timestamp_seconds <= time_range[1])
        )
    stmt = stmt.order_by(Detection.confidence.desc()).limit(top_k)

    rows = (await session.execute(stmt)).scalars().all()
    items = [{
        "detection_id": str(d.id),
        "frame_id": str(d.frame_id),
        "video_id": str(d.video_id),
        "class_name": d.class_name,
        "confidence": d.confidence,
        "bbox": d.bbox,
        "instance_id": d.instance_id,
    } for d in rows]

    by_class = {}
    by_instance = set()
    for it in items:
        by_class[it["class_name"]] = by_class.get(it["class_name"], 0) + 1
        if it["instance_id"] is not None:
            by_instance.add((it["video_id"], it["instance_id"]))

    summary = (
        f"get_object_detections(classes={classes}, min_conf={min_conf}): "
        f"{len(items)} detections, by_class={by_class}, "
        f"distinct_tracks={len(by_instance)}"
    )

    return ToolResult(
        model_summary=summary,
        ui_payload={"results": items},
        top_k_used=len(items),
    )
