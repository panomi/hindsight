"""get_frames_around_transcript — frames (and detections) shown while a phrase was spoken.

Composition tool.  Internally:
    1. Embed the query and find transcript segments that match (semantic +
       substring, same logic as search_transcript).
    2. For each match, pull frames in [start - pad, end + pad].
    3. Optionally restrict to frames containing detections of `classes`
       (defaults to ['person'] — the typical use case is "frames of the
       person who said X" for subject registration).

Without speaker diarisation we cannot single out the speaker.  The intended
flow is: agent calls this tool, gets candidate person detections, then offers
them to the user via `request_user_confirmation` mode='instances' so the user
identifies the actual speaker.
"""
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.embed import embed_text
from app.agent.schemas import ToolResult
from app.agent.tools._utils import empty_scope_result, scope_videos
from app.config import get_settings
from app.models import Detection, Frame, TranscriptSegment

settings = get_settings()

SCHEMA = {
    "name": "get_frames_around_transcript",
    "description": (
        "Find frames (and detections within them) shown while a phrase was "
        "being spoken. Use for questions like 'who is on screen when X is "
        "said?' or 'show me the person who said \"Y\"'. Composes "
        "search_transcript + get_object_detections internally — do NOT chain "
        "those manually for this purpose. Note: no speaker diarisation; if "
        "you need to identify the actual speaker, propose the returned "
        "person detections to the user via request_user_confirmation "
        "mode='instances'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Phrase to search for in transcripts."},
            "classes": {
                "type": "array", "items": {"type": "string"}, "default": ["person"],
                "description": "Detection classes to include in returned frames. Default ['person'].",
            },
            "padding_seconds": {
                "type": "number", "default": 1.5,
                "description": "Extend each transcript window by this on both sides.",
            },
            "video_ids": {"type": "array", "items": {"type": "string"}},
            "transcript_top_k": {
                "type": "integer", "default": 5, "maximum": 20,
                "description": "How many transcript matches to expand into frames. Keep small.",
            },
            "frames_per_window": {
                "type": "integer", "default": 6, "maximum": 20,
                "description": "Cap on detections returned per transcript window.",
            },
        },
        "required": ["query"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    query = params["query"]
    classes = params.get("classes") or ["person"]
    padding = float(params.get("padding_seconds", 1.5))
    video_ids = await scope_videos(session, investigation_id, params.get("video_ids", []))
    transcript_top_k = min(int(params.get("transcript_top_k", 5)), 20)
    frames_per_window = min(int(params.get("frames_per_window", 6)), 20)

    if not video_ids:
        return empty_scope_result("get_frames_around_transcript", query)

    # 1) Find transcript matches (semantic + substring fallback).
    qvec = embed_text(query)
    seg_stmt = (
        select(TranscriptSegment.id, TranscriptSegment.video_id,
               TranscriptSegment.text, TranscriptSegment.start_seconds,
               TranscriptSegment.end_seconds,
               TranscriptSegment.embedding.cosine_distance(qvec).label("distance"))
        .where(TranscriptSegment.video_id.in_(video_ids))
        .where(TranscriptSegment.embedding.is_not(None))
        .where(or_(TranscriptSegment.text.ilike(f"%{query}%"),
                   TranscriptSegment.id.is_not(None)))
        .order_by("distance")
        .limit(transcript_top_k)
    )
    segments = (await session.execute(seg_stmt)).all()
    if not segments:
        return ToolResult(
            model_summary=f"get_frames_around_transcript('{query[:60]}'): 0 transcript matches",
            ui_payload={"results": [], "query": query},
        )

    # 2) For each transcript window, fetch detections of `classes` whose frame
    #    timestamp falls in [start - pad, end + pad].  We do one query per
    #    segment to keep the SQL simple and the per-window cap honest.
    windows = []
    for seg in segments:
        win_start = float(seg.start_seconds) - padding
        win_end = float(seg.end_seconds) + padding

        det_stmt = (
            select(Detection, Frame)
            .join(Frame, Frame.id == Detection.frame_id)
            .where(and_(
                Detection.video_id == seg.video_id,
                Detection.class_name.in_(classes),
                Frame.timestamp_seconds >= win_start,
                Frame.timestamp_seconds <= win_end,
            ))
            .order_by(Detection.confidence.desc())
            .limit(frames_per_window)
        )
        rows = (await session.execute(det_stmt)).all()
        detections = [{
            "detection_id": str(d.id),
            "frame_id": str(d.frame_id),
            "video_id": str(d.video_id),
            "class_name": d.class_name,
            "confidence": d.confidence,
            "bbox": d.bbox,
            "instance_id": d.instance_id,
            "timestamp_seconds": f.timestamp_seconds,
        } for d, f in rows]

        windows.append({
            "segment_id": str(seg.id),
            "video_id": str(seg.video_id),
            "text": seg.text,
            "start_seconds": float(seg.start_seconds),
            "end_seconds": float(seg.end_seconds),
            "match_score": float(1.0 - (seg.distance or 1.0)),
            "detections": detections,
        })

    total_dets = sum(len(w["detections"]) for w in windows)
    distinct_tracks = {(w["video_id"], d["instance_id"])
                       for w in windows for d in w["detections"]
                       if d["instance_id"] is not None}

    summary = (
        f"get_frames_around_transcript('{query[:60]}'): {len(windows)} "
        f"transcript window(s), {total_dets} {','.join(classes)} detections, "
        f"{len(distinct_tracks)} distinct track(s)"
    )

    # Flatten into a `results` list as well so the UI's FrameGrid renders
    # naturally (it picks `results | events | matches`).
    flat_results = [d for w in windows for d in w["detections"]]

    return ToolResult(
        model_summary=summary,
        ui_payload={
            "results": flat_results,
            "windows": windows,
            "query": query,
            "classes": classes,
        },
        top_k_used=total_dets,
    )
