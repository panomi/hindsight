"""Open-vocabulary detection.

Two-stage to keep cost bounded:
  1. Cross-modal text→frame ANN to pick top-N candidate frames.
  2. Open-vocab detector run only on those candidate frames.

Results cached by (sha256(prompt), frame_id) in prompt_cache. In fake mode
we synthesise a bbox covering the upper-centre of each top-N frame so the
co_presence flow is exercisable end-to-end.

Implementation note: stage 2 currently uses Florence-2's OPEN_VOCABULARY_
DETECTION head, which does NOT emit confidence scores — we hard-code 0.8.
Treat the returned `score` as a presence flag, not a calibrated probability.
A future swap to SAM 3.1 / Grounding DINO would restore real scores.
"""
import hashlib
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.embed import embed_text
from app.agent.schemas import ToolResult
from app.agent.tools._utils import empty_scope_result, scope_videos
from app.config import get_settings
from app.models import Frame, PromptCache

settings = get_settings()

SCHEMA = {
    "name": "open_vocab_detect",
    "description": "Open-vocabulary text-prompted detection for concepts "
                   "outside the closed object class set. Use canonical noun "
                   "phrases ('person carrying red backpack') for cache-hit "
                   "rate. Internally pre-filters candidate frames via "
                   "cross-modal similarity before running detection. "
                   "Returned `score` is a presence flag, not a calibrated "
                   "probability — use it for ranking, not absolute thresholds.",
    "input_schema": {
        "type": "object",
        "properties": {
            "text_prompt": {"type": "string"},
            "video_id": {"type": "string"},
            "time_range": {
                "type": "array", "items": {"type": "number"},
                "minItems": 2, "maxItems": 2,
            },
            "candidate_top_k": {
                "type": "integer", "default": 20, "maximum": 50,
                "description": "How many candidate frames to run the detector on",
            },
        },
        "required": ["text_prompt"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    prompt: str = params["text_prompt"]
    time_range = params.get("time_range")
    top_k = min(int(params.get("candidate_top_k", 20)), settings.tool_result_top_k_default)

    # Investigation-scoped: agent may pass `video_id` to narrow further, but
    # we ALWAYS intersect with the investigation's collection so a guessed
    # video UUID from another investigation cannot be queried.
    explicit = [params["video_id"]] if params.get("video_id") else []
    video_ids = await scope_videos(session, investigation_id, explicit)
    if not video_ids:
        return empty_scope_result("open_vocab_detect", prompt)

    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    qvec = embed_text(prompt)

    # Stage 1: cross-modal candidate retrieval (SigLIP frame embeddings)
    stmt = (
        select(
            Frame.id, Frame.video_id, Frame.timestamp_seconds, Frame.filepath,
            Frame.siglip_embedding.cosine_distance(qvec).label("distance"),
        )
        .where(Frame.siglip_embedding.is_not(None))
        .where(Frame.video_id.in_(video_ids))
    )
    if time_range:
        stmt = stmt.where(and_(
            Frame.timestamp_seconds >= time_range[0],
            Frame.timestamp_seconds <= time_range[1],
        ))
    stmt = stmt.order_by("distance").limit(top_k)
    candidates = (await session.execute(stmt)).all()

    if not candidates:
        return ToolResult(
            model_summary=f"open_vocab_detect('{prompt[:60]}'): no candidate frames",
            ui_payload={"results": [], "prompt": prompt}, top_k_used=0,
        )

    # Stage 2: open-vocab detector — fake mode synthesises deterministic bboxes
    items = []
    for c in candidates:
        # Cache check
        cached = await session.scalar(
            select(PromptCache.payload).where(and_(
                PromptCache.prompt_hash == prompt_hash,
                PromptCache.frame_id == c.id,
                PromptCache.tool == "open_vocab_detect",
            ))
        )
        if cached is not None:
            items.append(cached)
            continue

        if settings.use_fake_ml:
            payload = {
                "frame_id": str(c.id), "video_id": str(c.video_id),
                "timestamp_seconds": c.timestamp_seconds, "filepath": c.filepath,
                "bbox": {"x1": 0.30, "y1": 0.20, "x2": 0.70, "y2": 0.80},
                "score": float(1.0 - (c.distance or 1.0)),
                "prompt": prompt,
            }
        else:
            payload = _real_detect(prompt, c)
            if payload is None:
                continue  # detector found nothing on this candidate frame
        items.append(payload)

        # Persist to cache
        await session.execute(insert(PromptCache).values(
            prompt_hash=prompt_hash, frame_id=c.id, tool="open_vocab_detect", payload=payload,
        ).on_conflict_do_nothing())
    await session.commit()

    # ── Surface frame_id + bbox in model_summary ──────────────────────────
    # The bboxes from this tool are the typical input to `ask_vision`
    # (cropping a license plate / schedule board / sign).  The agent cannot
    # chain that without seeing the bbox dicts, so we list each hit
    # compactly.  Note: bboxes are normalised (0..1) — same shape ask_vision
    # accepts directly, no reformatting needed.
    head = (
        f"open_vocab_detect('{prompt[:60]}'): {len(items)} frames, "
        f"score range [{min(i['score'] for i in items):.3f}, "
        f"{max(i['score'] for i in items):.3f}]"
    )
    max_lines = 12
    lines = []
    for it in items[:max_lines]:
        b = it["bbox"]
        lines.append(
            f"  [score={it['score']:.2f}] v={it['video_id'][:8]} "
            f"t={it['timestamp_seconds']:.1f}s frame={it['frame_id'][:8]} "
            f"bbox=[{b['x1']:.2f},{b['y1']:.2f},{b['x2']:.2f},{b['y2']:.2f}]"
        )
    more = f"\n  …{len(items) - max_lines} more not shown" if len(items) > max_lines else ""
    body = "\n" + "\n".join(lines) + more if lines else ""

    return ToolResult(
        model_summary=head + body,
        ui_payload={"results": items, "prompt": prompt},
        top_k_used=len(items),
    )


def _real_detect(prompt: str, candidate) -> dict | None:
    """Run Florence-2 OPEN_VOCABULARY_DETECTION on a single candidate frame.

    Florence-2 accepts dot-separated noun phrases as labels. We pass the full
    prompt as-is and take the highest-scoring detection.

    TODO: swap to SAM 3.1 once pytorch>=2.7 + sam3 package are available.
    """
    from PIL import Image

    from app.worker.ml import florence2_open_vocab_detect

    try:
        img = Image.open(candidate.filepath).convert("RGB")
    except Exception:
        return None

    detections = florence2_open_vocab_detect(img, prompt)
    if not detections:
        return None

    best = max(detections, key=lambda d: d["score"])
    return {
        "frame_id": str(candidate.id),
        "video_id": str(candidate.video_id),
        "timestamp_seconds": candidate.timestamp_seconds,
        "filepath": candidate.filepath,
        "bbox": best["bbox"],
        "score": best["score"],
        "label": best.get("label", prompt),
        "prompt": prompt,
        "all_instances": detections,
    }