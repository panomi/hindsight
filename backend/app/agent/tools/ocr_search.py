"""Search recognised on-screen text (license plates, signs, screens, papers).

Backed by RapidOCR at ingest time.  OCR is gated by caption text-carrier
flags so it only runs where readable text is plausibly visible.
"""
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.embed import embed_text
from app.agent.schemas import ToolResult
from app.agent.tools._utils import empty_scope_result, scope_videos
from app.config import get_settings
from app.models import Frame, OcrText

settings = get_settings()

SCHEMA = {
    "name": "search_ocr",
    "description": (
        "Search readable text recognised in the scene (license plates, signs, "
        "screens, paper, labels). Returns frames + bbox of the matched text. "
        "Note: text recognition runs **only** on frames whose scene "
        "description flagged a text carrier (sign/screen/paper/plate/label/"
        "newspaper/menu). Empty results may mean no carrier was flagged for "
        "those frames — not necessarily that no text exists. If an OCR query "
        "comes back empty for a known text-bearing scene, fall back to "
        "`open_vocab_detect` with the text-carrier word as the prompt to "
        "locate candidate frames."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "video_ids": {"type": "array", "items": {"type": "string"}},
            "top_k": {"type": "integer", "default": 20, "maximum": 50},
        },
        "required": ["query"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    query = params["query"]
    video_ids = await scope_videos(session, investigation_id, params.get("video_ids", []))
    if not video_ids:
        return empty_scope_result("search_ocr", query)
    top_k = min(int(params.get("top_k", 20)), settings.tool_result_top_k_default)

    # OCR search prefers exact / fuzzy substring (license plates rarely embed well).
    # We do both: substring filter first, then ANN re-rank if many hits.
    stmt = (
        select(
            OcrText.id, OcrText.frame_id, OcrText.text, OcrText.bbox, OcrText.confidence,
            Frame.video_id, Frame.timestamp_seconds, Frame.filepath,
        )
        .join(Frame, Frame.id == OcrText.frame_id)
        .where(OcrText.text.ilike(f"%{query}%"))
        .where(Frame.video_id.in_(video_ids))
        .order_by(OcrText.confidence.desc())
        .limit(top_k)
    )

    rows = (await session.execute(stmt)).all()
    if not rows:
        # Fall back to embedding similarity
        qvec = embed_text(query)
        stmt2 = (
            select(
                OcrText.id, OcrText.frame_id, OcrText.text, OcrText.bbox, OcrText.confidence,
                Frame.video_id, Frame.timestamp_seconds, Frame.filepath,
                OcrText.embedding.cosine_distance(qvec).label("distance"),
            )
            .join(Frame, Frame.id == OcrText.frame_id)
            .where(OcrText.embedding.is_not(None))
            .where(Frame.video_id.in_(video_ids))
            .order_by("distance")
            .limit(top_k)
        )
        rows = (await session.execute(stmt2)).all()

    items = [{
        "ocr_id": str(r.id),
        "frame_id": str(r.frame_id),
        "video_id": str(r.video_id),
        "text": r.text,
        "bbox": r.bbox,
        "confidence": r.confidence,
        "timestamp_seconds": r.timestamp_seconds,
        "filepath": r.filepath,
    } for r in rows]

    return ToolResult(
        model_summary=f"search_ocr('{query[:60]}'): {len(items)} text hits",
        ui_payload={"results": items, "query": query},
        top_k_used=len(items),
    )
