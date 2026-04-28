"""Group frame hits into contiguous time windows (events).

Uses shot boundaries when available to keep cluster edges aligned with
real scene changes, falling back to a fixed gap.
"""
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import ToolResult
from app.models import Frame, Shot

def _fmt(s: float) -> str:
    s = int(s)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


SCHEMA = {
    "name": "temporal_cluster",
    "description": "Group frame hits into time windows (events), sorted by "
                   "relevance (highest-scoring event first). Pass scores from "
                   "search tools so events are ranked by evidence quality, not time.",
    "input_schema": {
        "type": "object",
        "properties": {
            "frame_ids": {"type": "array", "items": {"type": "string"}},
            "scores": {
                "type": "array", "items": {"type": "number"},
                "description": "Relevance score per frame_id (parallel list). "
                               "Parse from frame_ids_with_scores in tool results.",
            },
            "gap_sec": {"type": "number", "default": 3.0,
                        "description": "Frames within this many seconds form one event"},
            "relative_min": {
                "type": "number", "default": 0.70,
                "description": "Keep frames whose score >= max_score * relative_min. "
                               "Adapts automatically to each model's score range — "
                               "0.70 means drop anything scoring below 70%% of the best hit.",
            },
        },
        "required": ["frame_ids"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    raw_ids = params["frame_ids"]
    raw_scores: list[float] = params.get("scores") or []
    gap_sec = float(params.get("gap_sec", 3.0))
    relative_min = float(params.get("relative_min", 0.70))

    # Build score lookup (frame_id str → score); default 1.0 if not provided
    if raw_scores and len(raw_scores) == len(raw_ids):
        score_map: dict[str, float] = {fid: float(s) for fid, s in zip(raw_ids, raw_scores)}
    else:
        score_map = {fid: 1.0 for fid in raw_ids}

    # Relative threshold: keep frames >= relative_min * max_score
    # This adapts to any model's score range automatically.
    max_score_overall = max(score_map.values()) if score_map else 1.0
    threshold = max_score_overall * relative_min
    filtered_ids = [fid for fid in raw_ids if score_map.get(fid, 1.0) >= threshold]
    pruned = len(raw_ids) - len(filtered_ids)

    if not filtered_ids:
        return ToolResult(
            model_summary=(
                f"temporal_cluster: 0 frames after relative filter "
                f"(threshold={threshold:.3f} = {relative_min:.0%} of best={max_score_overall:.3f}, "
                f"pruned={pruned}). Try relative_min=0.5 to widen the net."
            ),
            ui_payload={"events": []},
        )

    frames = (await session.execute(
        select(Frame.id, Frame.video_id, Frame.timestamp_seconds, Frame.shot_id)
        .where(Frame.id.in_([UUID(f) for f in filtered_ids]))
        .order_by(Frame.video_id, Frame.timestamp_seconds)
    )).all()
    if not frames:
        return ToolResult(model_summary="temporal_cluster: no frames found", ui_payload={"events": []})

    # Cluster within video, breaking on gap > gap_sec OR shot boundary change
    events: list[dict] = []
    cur_video = None
    cur_event: dict | None = None
    last_ts = None
    last_shot = None

    for f in frames:
        fid_str = str(f.id)
        score = score_map.get(fid_str, 1.0)
        new_event = (
            cur_event is None
            or f.video_id != cur_video
            or (last_ts is not None and (f.timestamp_seconds - last_ts) > gap_sec)
            or (f.shot_id is not None and last_shot is not None and f.shot_id != last_shot)
        )
        if new_event:
            cur_event = {
                "video_id": str(f.video_id),
                "start_seconds": f.timestamp_seconds,
                "end_seconds": f.timestamp_seconds,
                "frame_count": 0,
                "frame_ids": [],
                "max_score": 0.0,
                # Representative frame (highest-scoring within the event) — set
                # so the UI can render a real thumbnail rather than a "frame"
                # placeholder.  Updated as we accumulate.
                "frame_id": fid_str,
                "timestamp_seconds": f.timestamp_seconds,
            }
            events.append(cur_event)
            cur_video = f.video_id
        cur_event["end_seconds"] = max(cur_event["end_seconds"], f.timestamp_seconds)
        cur_event["frame_count"] += 1
        cur_event["frame_ids"].append(fid_str)
        if score > cur_event["max_score"]:
            cur_event["max_score"] = score
            cur_event["frame_id"] = fid_str
            cur_event["timestamp_seconds"] = f.timestamp_seconds
        last_ts = f.timestamp_seconds
        last_shot = f.shot_id

    # Sort by max_score descending — best evidence first
    events.sort(key=lambda e: e["max_score"], reverse=True)

    # Add a human-readable label so the FrameGrid text row shows the time
    # range instead of the generic "result" fallback.
    for e in events:
        e["text"] = (
            f"{_fmt(e['start_seconds'])}–{_fmt(e['end_seconds'])} "
            f"({e['frame_count']}f)"
        )

    event_lines = [
        f"  [{i+1}] {_fmt(e['start_seconds'])}–{_fmt(e['end_seconds'])} "
        f"score={e['max_score']:.3f} "
        f"({e['frame_count']} frame{'s' if e['frame_count'] != 1 else ''}, "
        f"video_id={e['video_id']})"
        for i, e in enumerate(events)
    ]
    pruned_note = (
        f" ({pruned} frames pruned below {threshold:.3f} = {relative_min:.0%} of best)"
        if pruned else ""
    )
    return ToolResult(
        model_summary=(
            f"temporal_cluster: {len(filtered_ids)} frames → {len(events)} events"
            f"{pruned_note}, ranked by score:\n"
            + "\n".join(event_lines)
        ),
        ui_payload={"events": events},
        top_k_used=len(events),
    )
