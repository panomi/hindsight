"""ask_vision — visual question-answering on specific frames.

Lets the agent point the vision-language model at one or more frames and ask
a free-form, frame-specific question (e.g. "what time does the Chicago flight
depart?", "what is the license plate?", "is the door open or closed?").
Optional bbox crops the input to a region of interest, which dramatically
improves accuracy on dense / small / stylised text — schedule boards, license
plates, signs, menus, screens.

Strict separation from `caption_frames`:
  * `caption_frames` returns the *stored* ingest description (one fixed prompt,
    paragraph-form scene summary) and is for retrieval / context.
  * `ask_vision` is *Q&A* — custom prompt per call, short factual answer, no
    persistence to the captions table, cached only in `prompt_cache`.

Cost discipline:
  * Hard cap: 6 frames per call.  VLM is ~0.5-2s per frame on the GPU;
    caps avoid surprise multi-second waits.
  * Per-investigation cache keyed by (sha256(question + bbox_json), frame_id)
    so re-asks within a session are free.
  * Bbox crop on the client side keeps the input size reasonable even for
    1080p sources.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import ToolResult
from app.agent.tools._utils import empty_scope_result, scope_videos
from app.config import get_settings
from app.models import Frame, PromptCache, Video

settings = get_settings()


SCHEMA = {
    "name": "ask_vision",
    "description": (
        "Ask the vision-language model a free-form question about specific frames. "
        "Use this when OCR found the entity but missed the data field you need "
        "(e.g. 'CHICAGO' on a schedule board but no time), when stored captions "
        "are too generic, or when you need to read tabular / dense / stylised "
        "text the OCR pass missed. Returns ONE short answer per frame; does NOT "
        "store anything. Cap 6 frames per call. Pass `bbox_xyxy` (normalised "
        "0..1) to crop the input — strongly recommended when you have a bbox "
        "from `open_vocab_detect` or a `Detection` (e.g. zoom in on the schedule "
        "board, the license plate, the sign). The model returns 'unknown' when "
        "the answer is not visible — do not invent."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "frame_ids": {
                "type": "array", "items": {"type": "string"},
                "minItems": 1, "maxItems": 6,
                "description": "Frames to inspect. Cap 6 per call.",
            },
            "question": {
                "type": "string",
                "description": (
                    "Concrete, focused question. Good: 'what time does the "
                    "flight to Chicago depart?', 'what is written on the "
                    "newspaper headline?'. Bad: 'describe the scene' (use "
                    "caption_frames). Bad: 'what should I do here' (not a "
                    "visual question)."
                ),
            },
            "bbox_xyxy": {
                "type": "object",
                "description": (
                    "Optional crop region as normalised (0..1) {x1,y1,x2,y2}. "
                    "Same dict shape returned by open_vocab_detect.bbox and "
                    "Detection.bbox. Applied to every frame in this call — for "
                    "per-frame bboxes, make separate calls."
                ),
                "properties": {
                    "x1": {"type": "number"}, "y1": {"type": "number"},
                    "x2": {"type": "number"}, "y2": {"type": "number"},
                },
            },
            "max_tokens": {
                "type": "integer", "default": 96, "maximum": 256,
                "description": "Cap on answer length. Default 96 — enough for a phrase or sentence.",
            },
        },
        "required": ["frame_ids", "question"],
    },
}


_MAX_FRAMES = 6


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    raw_frame_ids: list[str] = list(params.get("frame_ids") or [])[:_MAX_FRAMES]
    question: str = (params.get("question") or "").strip()
    bbox: dict | None = params.get("bbox_xyxy")
    max_tokens: int = min(int(params.get("max_tokens", 96)), 256)

    if not raw_frame_ids:
        return ToolResult(
            model_summary="ask_vision: no frame_ids provided",
            ui_payload={"results": [], "question": question},
            top_k_used=0,
        )
    if not question:
        return ToolResult(
            model_summary="ask_vision: empty question",
            ui_payload={"results": [], "question": question},
            top_k_used=0,
        )

    # Normalise bbox once; reject malformed silently rather than 500.
    norm_bbox = _normalise_bbox(bbox)

    # Scope guard: we accept frame_ids the agent obtained from earlier tools
    # but we re-validate every one against the investigation's collection so
    # a guessed UUID from another collection cannot leak pixels.
    scope = await scope_videos(session, investigation_id, [])
    if not scope:
        return empty_scope_result("ask_vision", question)

    try:
        frame_uuids = [UUID(fid) for fid in raw_frame_ids]
    except ValueError:
        return ToolResult(
            model_summary="ask_vision: invalid frame_id format",
            ui_payload={"results": [], "question": question},
            top_k_used=0,
        )

    rows = (await session.execute(
        select(
            Frame.id, Frame.video_id, Frame.timestamp_seconds, Frame.filepath,
        )
        .join(Video, Video.id == Frame.video_id)
        .where(Frame.id.in_(frame_uuids))
        .where(Frame.video_id.in_(scope))
    )).all()

    if not rows:
        return ToolResult(
            model_summary=(
                f"ask_vision('{question[:60]}'): 0 frames in scope "
                f"({len(frame_uuids)} requested)"
            ),
            ui_payload={"results": [], "question": question},
            top_k_used=0,
        )

    # Preserve the order the agent asked in (ANN scoring is order-sensitive).
    by_id = {r.id: r for r in rows}
    ordered = [by_id[uid] for uid in frame_uuids if uid in by_id]

    prompt_hash = hashlib.sha256(_prompt_key(question, norm_bbox).encode()).hexdigest()

    # ── Cache lookup ───────────────────────────────────────────────────────
    cached_payloads: dict[UUID, dict] = {}
    cache_rows = (await session.execute(
        select(PromptCache.frame_id, PromptCache.payload).where(and_(
            PromptCache.prompt_hash == prompt_hash,
            PromptCache.tool == "ask_vision",
            PromptCache.frame_id.in_([r.id for r in ordered]),
        ))
    )).all()
    for fid, payload in cache_rows:
        cached_payloads[fid] = payload

    misses = [r for r in ordered if r.id not in cached_payloads]

    # ── Real or fake VLM ───────────────────────────────────────────────────
    fresh: list[dict] = []
    if misses:
        if settings.use_fake_ml:
            fresh = [_fake_answer(r, question, norm_bbox) for r in misses]
        else:
            fresh = await _real_vqa(misses, question, norm_bbox, max_tokens)

    # Persist new answers to the cache.
    for payload in fresh:
        await session.execute(insert(PromptCache).values(
            prompt_hash=prompt_hash,
            frame_id=UUID(payload["frame_id"]),
            tool="ask_vision",
            payload=payload,
        ).on_conflict_do_nothing())
    if fresh:
        await session.commit()

    # ── Merge in input order ───────────────────────────────────────────────
    fresh_by_id: dict[UUID, dict] = {UUID(p["frame_id"]): p for p in fresh}
    items: list[dict] = []
    for r in ordered:
        items.append(cached_payloads.get(r.id) or fresh_by_id.get(r.id) or _empty_answer(r, question))

    # ── Surface answers in model_summary ───────────────────────────────────
    head = (
        f"ask_vision('{question[:60]}'): {len(items)} frame(s) answered"
        + (f", bbox={_short_bbox(norm_bbox)}" if norm_bbox else "")
    )
    lines = []
    for it in items:
        ans = (it.get("answer") or "").strip().replace("\n", " ")
        if len(ans) > 240:
            ans = ans[:239] + "…"
        cached_tag = " (cached)" if it.get("frame_id") in {str(fid) for fid in cached_payloads} else ""
        lines.append(
            f"  v={it['video_id'][:8]} t={it['timestamp_seconds']:.1f}s "
            f"frame={it['frame_id'][:8]}{cached_tag}: \"{ans}\""
        )
    body = "\n" + "\n".join(lines) if lines else ""

    return ToolResult(
        model_summary=head + body,
        ui_payload={"results": items, "question": question, "bbox_xyxy": norm_bbox},
        top_k_used=len(items),
    )


# ── Helpers ────────────────────────────────────────────────────────────────


def _normalise_bbox(bbox: Any) -> dict | None:
    """Accept {x1,y1,x2,y2} dict; clamp to [0,1]; reject if degenerate."""
    if not isinstance(bbox, dict):
        return None
    try:
        x1, y1, x2, y2 = (float(bbox["x1"]), float(bbox["y1"]),
                          float(bbox["x2"]), float(bbox["y2"]))
    except (KeyError, TypeError, ValueError):
        return None
    x1, x2 = sorted([max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))])
    y1, y2 = sorted([max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))])
    if (x2 - x1) < 0.02 or (y2 - y1) < 0.02:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _short_bbox(b: dict) -> str:
    return f"[{b['x1']:.2f},{b['y1']:.2f},{b['x2']:.2f},{b['y2']:.2f}]"


def _prompt_key(question: str, bbox: dict | None) -> str:
    """Stable cache key: question + bbox.  Different bbox => different cache row."""
    return json.dumps({"q": question, "b": bbox}, sort_keys=True)


def _empty_answer(r: Any, question: str) -> dict:
    return {
        "frame_id": str(r.id),
        "video_id": str(r.video_id),
        "timestamp_seconds": float(r.timestamp_seconds),
        "filepath": r.filepath,
        "answer": "unknown",
        "question": question,
    }


def _fake_answer(r: Any, question: str, bbox: dict | None) -> dict:
    """Deterministic placeholder so end-to-end flows are testable without a GPU."""
    return {
        "frame_id": str(r.id),
        "video_id": str(r.video_id),
        "timestamp_seconds": float(r.timestamp_seconds),
        "filepath": r.filepath,
        "answer": f"[fake] re: '{question[:40]}' on frame at {r.timestamp_seconds:.1f}s",
        "question": question,
        "bbox_xyxy": bbox,
    }


async def _real_vqa(
    rows: list[Any], question: str, bbox: dict | None, max_tokens: int,
) -> list[dict]:
    """Real-mode VLM call.  Crops to bbox if provided, batches the forward pass."""
    from PIL import Image

    from app.worker.ml import qwen_vl_vqa_batch

    images: list = []
    valid_rows: list[Any] = []
    for r in rows:
        try:
            img = Image.open(r.filepath).convert("RGB")
        except Exception:
            continue
        if bbox is not None:
            img = _crop_norm(img, bbox)
        images.append(img)
        valid_rows.append(r)

    if not images:
        return []

    answers = qwen_vl_vqa_batch(
        images=images,
        questions=[question] * len(images),
        max_new_tokens=max_tokens,
    )

    return [{
        "frame_id": str(r.id),
        "video_id": str(r.video_id),
        "timestamp_seconds": float(r.timestamp_seconds),
        "filepath": r.filepath,
        "answer": ans.strip(),
        "question": question,
        "bbox_xyxy": bbox,
    } for r, ans in zip(valid_rows, answers, strict=True)]


def _crop_norm(img, bbox: dict):
    """Crop a PIL image to a normalised bbox; expand by 5% margin so context isn't cut."""
    w, h = img.width, img.height
    margin = 0.05
    x1 = max(0.0, bbox["x1"] - margin) * w
    y1 = max(0.0, bbox["y1"] - margin) * h
    x2 = min(1.0, bbox["x2"] + margin) * w
    y2 = min(1.0, bbox["y2"] + margin) * h
    return img.crop((int(x1), int(y1), int(x2), int(y2)))
