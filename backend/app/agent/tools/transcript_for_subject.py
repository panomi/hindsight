"""get_transcript_for_subject — every transcript line spoken while a subject is on screen.

This is a *composition* tool: instead of asking the agent to chain
`search_instances_by_subject` → ferry timestamps → `search_transcript`, we do
the join inside the database.  The model_summary contains a few representative
quotes so the agent has something to quote in its reply without a second call.

Join chain:
    SubjectInstance (subject_id → video_id, instance_id)
      → Detection (video_id, instance_id → frame_id)
      → Frame (frame_id → timestamp_seconds)
      → TranscriptSegment overlapping [t - pad, t + pad]
"""
from uuid import UUID

from sqlalchemy import and_, or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import ToolResult
from app.agent.tools._utils import validate_subject
from app.config import get_settings
from app.models import Detection, Frame, SubjectInstance, TranscriptSegment

settings = get_settings()

SCHEMA = {
    "name": "get_transcript_for_subject",
    "description": (
        "Return every transcript segment spoken while a registered Subject is "
        "on screen. Use this for questions like 'what did this person say?', "
        "'any notable quotes from Subject A?', or 'what was being discussed "
        "while X was visible?'. Composes SubjectInstance → Detection → Frame "
        "→ TranscriptSegment internally, so you do NOT need to chain "
        "search_instances_by_subject + search_transcript yourself. Note: we "
        "do not have speaker diarisation — segments are 'spoken in the "
        "subject's presence', not necessarily 'spoken by the subject'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subject_id": {"type": "string"},
            "padding_seconds": {
                "type": "number",
                "default": 1.5,
                "description": "Extend each appearance window by this many seconds on both sides "
                               "to catch speech that overlaps the subject's entry/exit.",
            },
            "video_ids": {
                "type": "array", "items": {"type": "string"},
                "description": "Optional video filter; defaults to all ready videos in scope.",
            },
            "top_k": {"type": "integer", "default": 50, "maximum": 100},
        },
        "required": ["subject_id"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    subject_id = UUID(params["subject_id"])
    padding = float(params.get("padding_seconds", 1.5))
    video_filter = [UUID(v) for v in params.get("video_ids", [])] or None
    top_k = min(int(params.get("top_k", 50)), 100)

    subject = await validate_subject(session, subject_id, investigation_id)
    if subject is None:
        return ToolResult(
            model_summary=f"get_transcript_for_subject({subject_id}): subject not found",
            ui_payload={"results": [], "subject_id": str(subject_id)},
            error="subject_not_in_scope",
        )

    # 1) Pull (video_id, instance_id) pairs for this subject — these are the
    #    tracks user-confirmed at registration plus any cross-video matches
    #    pulled in by search_instances_by_subject.
    pairs_stmt = select(SubjectInstance.video_id, SubjectInstance.instance_id) \
        .where(SubjectInstance.subject_id == subject_id)
    if video_filter:
        pairs_stmt = pairs_stmt.where(SubjectInstance.video_id.in_(video_filter))
    pairs = (await session.execute(pairs_stmt)).all()
    if not pairs:
        return ToolResult(
            model_summary=(
                f"get_transcript_for_subject({subject.label}): subject has no "
                f"matched (video, instance) pairs yet — call "
                f"search_instances_by_subject first to propagate matches."
            ),
            ui_payload={"results": [], "subject_id": str(subject_id), "label": subject.label},
        )
    pair_keys = [(v, iid) for v, iid in pairs]

    # 2) Fetch the timestamp of every frame where this subject appears.
    frame_times_stmt = (
        select(Detection.video_id, Frame.timestamp_seconds)
        .join(Frame, Frame.id == Detection.frame_id)
        .where(tuple_(Detection.video_id, Detection.instance_id).in_(pair_keys))
    )
    times_by_video: dict[UUID, list[float]] = {}
    for video_id, ts in (await session.execute(frame_times_stmt)).all():
        times_by_video.setdefault(video_id, []).append(float(ts))
    if not times_by_video:
        return ToolResult(
            model_summary=(
                f"get_transcript_for_subject({subject.label}): no frames found for "
                f"subject's matched tracks (data inconsistency — re-run "
                f"search_instances_by_subject)."
            ),
            ui_payload={"results": [], "subject_id": str(subject_id), "label": subject.label},
        )

    # 3) Collapse per-video timestamps into [start, end] intervals padded by
    #    `padding`.  Adjacent appearances within 2*padding merge so we don't
    #    emit a thousand tiny windows for a long shot.
    intervals_by_video: dict[UUID, list[tuple[float, float]]] = {}
    for video_id, times in times_by_video.items():
        times.sort()
        merged: list[tuple[float, float]] = []
        cur_start = times[0] - padding
        cur_end = times[0] + padding
        for t in times[1:]:
            if t - padding <= cur_end:
                cur_end = max(cur_end, t + padding)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = t - padding, t + padding
        merged.append((cur_start, cur_end))
        intervals_by_video[video_id] = merged

    # 4) Pull transcript segments overlapping any of those intervals.
    overlap_clauses = []
    for video_id, intervals in intervals_by_video.items():
        per_video = [
            and_(TranscriptSegment.start_seconds <= e,
                 TranscriptSegment.end_seconds >= s)
            for s, e in intervals
        ]
        overlap_clauses.append(and_(
            TranscriptSegment.video_id == video_id,
            or_(*per_video),
        ))
    seg_stmt = (
        select(TranscriptSegment)
        .where(or_(*overlap_clauses))
        .order_by(TranscriptSegment.video_id, TranscriptSegment.start_seconds)
        .limit(top_k)
    )
    segments = (await session.execute(seg_stmt)).scalars().all()

    items = [{
        "segment_id": str(s.id),
        "video_id": str(s.video_id),
        "text": s.text,
        "start_seconds": s.start_seconds,
        "end_seconds": s.end_seconds,
    } for s in segments]

    # Surface a couple of representative quotes in the model_summary so the
    # agent can quote without a second call.
    preview = " | ".join(
        f"[{int(s.start_seconds // 60):02d}:{int(s.start_seconds % 60):02d}] "
        f"{s.text[:80]}"
        for s in segments[:3]
    )
    summary = (
        f"get_transcript_for_subject({subject.label}): {len(items)} segments "
        f"across {len({i['video_id'] for i in items})} video(s), "
        f"{sum(len(v) for v in intervals_by_video.values())} appearance window(s)"
        + (f". First lines: {preview}" if preview else "")
    )

    return ToolResult(
        model_summary=summary,
        ui_payload={
            "results": items,
            "subject_id": str(subject_id),
            "label": subject.label,
            "appearance_windows": {
                str(v): [[round(s, 2), round(e, 2)] for s, e in intervals]
                for v, intervals in intervals_by_video.items()
            },
        },
        top_k_used=len(items),
    )
