"""Frame-level SigLIP ANN search."""
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.embed import embed_text
from app.agent.schemas import ToolResult
from app.agent.tools._utils import empty_scope_result, scope_videos
from app.config import get_settings
from app.models import Frame

settings = get_settings()

SCHEMA = {
    "name": "search_visual_embeddings",
    "description": "Frame-level semantic search via SigLIP. Use for natural-language queries that "
                   "describe visual content of a scene. Returns ranked frames; never raw bbox data.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language description"},
            "top_k": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50},
            "video_ids": {
                "type": "array", "items": {"type": "string"},
                "description": "Optional: restrict to these video UUIDs",
            },
        },
        "required": ["query"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    query: str = params["query"]
    top_k = min(int(params.get("top_k", 20)), settings.tool_result_top_k_default)
    video_ids = await scope_videos(session, investigation_id, params.get("video_ids", []))
    if not video_ids:
        return empty_scope_result("search_visual_embeddings", query)

    qvec = embed_text(query)

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
    if not rows:
        return ToolResult(
            model_summary=f"search_visual_embeddings('{query[:60]}'): 0 results",
            ui_payload={"results": []}, top_k_used=0,
        )

    all_items = [{
        "frame_id": str(r.id), "video_id": str(r.video_id),
        "timestamp_seconds": r.timestamp_seconds,
        "filepath": r.filepath,
        "score": float(1.0 - (r.distance or 1.0)),
    } for r in rows]

    # Drop results with non-positive scores — cosine distance ≥ 1.0 means the
    # embedding is orthogonal or opposite to the query (worse than random noise).
    positive = [i for i in all_items if i["score"] > 0]
    if not positive:
        return ToolResult(
            model_summary=f"search_visual_embeddings('{query[:60]}'): 0 results above noise floor",
            ui_payload={"results": []}, top_k_used=0,
        )

    # Relative filter: keep only results within 70% of the best score.
    # This adapts to each model's score range automatically.
    best = max(i["score"] for i in positive)
    items = [i for i in positive if i["score"] >= best * 0.70]

    score_min = min(i["score"] for i in items)
    score_max = max(i["score"] for i in items)
    pruned = len(all_items) - len(items)

    scored_csv = ", ".join(f"{i['frame_id']}:{i['score']:.3f}" for i in items[:20])
    pruned_note = f" ({pruned} below noise floor or relative threshold pruned)" if pruned else ""
    return ToolResult(
        model_summary=(
            f"search_visual_embeddings('{query[:60]}'): {len(items)} hits{pruned_note}, "
            f"score range [{score_min:.3f}, {score_max:.3f}], "
            f"across {len({i['video_id'] for i in items})} video(s)."
            f"\nframe_ids_with_scores: [{scored_csv}]"
        ),
        ui_payload={"results": items, "query": query},
        top_k_used=len(items),
    )
