"""Image-based search at frame OR box (detection crop) level.

scope='frame' compares the query embedding to frames.siglip_embedding.
scope='box'   compares to detections.box_embedding (the appearance signal
              for instance retrieval). Optionally filtered by class_filter.
"""
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.embed import embed_image_b64
from app.agent.schemas import ToolResult
from app.agent.tools._utils import empty_scope_result, scope_videos
from app.config import get_settings
from app.models import Detection, Frame

settings = get_settings()

SCHEMA = {
    "name": "search_by_image",
    "description": "Image-based ANN search. scope='frame' matches whole frames; "
                   "scope='box' matches per-detection crops (use this for "
                   "'find this person/vehicle' queries). Optional class_filter "
                   "narrows box matches (e.g. 'person').",
    "input_schema": {
        "type": "object",
        "properties": {
            "image_b64": {"type": "string", "description": "Base64-encoded reference image"},
            "top_k": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50},
            "video_ids": {"type": "array", "items": {"type": "string"}},
            "scope": {"type": "string", "enum": ["frame", "box"], "default": "box"},
            "class_filter": {"type": "string", "description": "Optional class name (box scope only)"},
        },
        "required": ["image_b64"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    image_b64: str = params["image_b64"]
    top_k = min(int(params.get("top_k", 20)), settings.tool_result_top_k_default)
    video_ids = await scope_videos(session, investigation_id, params.get("video_ids", []))
    if not video_ids:
        return empty_scope_result(f"search_by_image[{params.get('scope', 'box')}]")
    scope = params.get("scope", "box")
    class_filter = params.get("class_filter")

    qvec = embed_image_b64(image_b64)

    if scope == "frame":
        stmt = (
            select(
                Frame.id, Frame.video_id, Frame.timestamp_seconds, Frame.filepath,
                Frame.siglip_embedding.cosine_distance(qvec).label("distance"),
            )
            .where(Frame.siglip_embedding.is_not(None))
            .where(Frame.video_id.in_(video_ids))
            .order_by("distance")
            .limit(top_k)
        )
        rows = (await session.execute(stmt)).all()
        items = [{
            "frame_id": str(r.id), "video_id": str(r.video_id),
            "timestamp_seconds": r.timestamp_seconds,
            "filepath": r.filepath,
            "score": float(1.0 - (r.distance or 1.0)),
        } for r in rows]
    else:
        stmt = (
            select(
                Detection.id, Detection.frame_id, Detection.video_id,
                Detection.class_name, Detection.bbox, Detection.instance_id,
                Detection.box_embedding.cosine_distance(qvec).label("distance"),
            )
            .where(Detection.box_embedding.is_not(None))
            .where(Detection.video_id.in_(video_ids))
        )
        if class_filter:
            stmt = stmt.where(Detection.class_name == class_filter)
        stmt = stmt.order_by("distance").limit(top_k)
        rows = (await session.execute(stmt)).all()
        items = [{
            "detection_id": str(r.id),
            "frame_id": str(r.frame_id),
            "video_id": str(r.video_id),
            "class_name": r.class_name,
            "bbox": r.bbox,
            "instance_id": r.instance_id,
            "score": float(1.0 - (r.distance or 1.0)),
        } for r in rows]

    if not items:
        return ToolResult(
            model_summary=f"search_by_image(scope={scope}): 0 results",
            ui_payload={"results": [], "scope": scope}, top_k_used=0,
        )

    instance_ids = sorted({i["instance_id"] for i in items if "instance_id" in i and i["instance_id"] is not None})
    summary = (
        f"search_by_image(scope={scope}, class={class_filter or '*'}): {len(items)} hits, "
        f"score range [{min(i['score'] for i in items):.3f}, {max(i['score'] for i in items):.3f}]"
    )
    if instance_ids:
        summary += f", spans instance_ids {instance_ids[:5]}{'…' if len(instance_ids) > 5 else ''}"

    return ToolResult(
        model_summary=summary,
        ui_payload={"results": items, "scope": scope, "class_filter": class_filter},
        top_k_used=len(items),
    )
