"""On-demand caption retrieval / densification.

Returns already-stored captions from ingest first (zero inference cost).
Only runs vision-language model inference on frames that have no caption yet.
"""
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import ToolResult
from app.config import get_settings
from app.models import Caption, Detection, Frame
from app.worker.util import deterministic_unit_vector

settings = get_settings()

SCHEMA = {
    "name": "caption_frames",
    "description": "Get scene descriptions for a specific list of frames. "
                   "Returns stored descriptions from ingest where available; "
                   "generates on-demand for any uncaptioned frames. "
                   "Bounded — avoid more than ~20 frames at a time.",
    "input_schema": {
        "type": "object",
        "properties": {
            "frame_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
        },
        "required": ["frame_ids"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    frame_ids = [UUID(f) for f in params["frame_ids"]][:30]

    # ── 1. Return existing captions from DB (free) ────────────────────────────
    existing_rows = (await session.execute(
        select(Caption.frame_id, Caption.text, Caption.source)
        .where(Caption.frame_id.in_(frame_ids))
    )).all()
    existing: dict[UUID, dict] = {
        r.frame_id: {"frame_id": str(r.frame_id), "text": r.text, "source": r.source}
        for r in existing_rows
    }

    # ── 2. Only run inference on uncaptioned frames ───────────────────────────
    uncaptioned_ids = [fid for fid in frame_ids if fid not in existing]
    items = list(existing.values())
    inferred = 0

    if uncaptioned_ids:
        frames = (await session.execute(
            select(Frame).where(Frame.id.in_(uncaptioned_ids))
        )).scalars().all()

        for f in frames:
            if settings.use_fake_ml:
                classes = [d.class_name for d in (await session.execute(
                    select(Detection).where(Detection.frame_id == f.id)
                )).scalars().all()]
                text = ("Scene contains " + ", ".join(sorted(set(classes))) + "."
                        if classes else "Empty scene.")
                source = "fake"
                cap = Caption(
                    id=uuid4(), shot_id=f.shot_id, frame_id=f.id,
                    text=text, source=source,
                    # 384-dim to match the new BGE-sized column.
                    embedding=deterministic_unit_vector(text.encode() + f.id.bytes, dim=384),
                )
            else:
                from PIL import Image

                from app.worker.ml import bge_encode_text, qwen_vl_caption
                try:
                    img = Image.open(f.filepath).convert("RGB")
                except Exception:
                    continue
                text = qwen_vl_caption(img)
                source = "vlm"
                # BGE for text-text retrieval (matches ingest path).
                emb = bge_encode_text([text])[0]
                cap = Caption(
                    id=uuid4(), shot_id=f.shot_id, frame_id=f.id,
                    text=text, source=source,
                    embedding=[float(x) for x in emb],
                )
            session.add(cap)
            items.append({"frame_id": str(f.id), "text": text, "source": source})
            inferred += 1

        await session.commit()

    cached = len(existing)
    return ToolResult(
        model_summary=(
            f"caption_frames({len(items)} frames): "
            f"{cached} from ingest, {inferred} generated on-demand"
        ),
        ui_payload={"results": items},
        top_k_used=len(items),
    )