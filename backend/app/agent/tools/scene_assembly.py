"""Build a structured event timeline for a Subject in a video.

Uses the matched (subject, video, instance_id) tuples from subject_instances
plus the within-video tracker continuity to identify enter / dwell / exit
moments, then attaches the nearest captions/OCR for context.
"""
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import ToolResult
from app.agent.tools._utils import validate_subject
from app.models import Caption, Detection, Frame, OcrText, SubjectInstance

SCHEMA = {
    "name": "scene_assembly",
    "description": "Assemble a structured event timeline (entered, dwelled, "
                   "exited) for a Subject in a specific video, with nearby "
                   "captions / OCR context.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subject_id": {"type": "string"},
            "video_id": {"type": "string"},
        },
        "required": ["subject_id", "video_id"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    subject_id = UUID(params["subject_id"])
    video_id = UUID(params["video_id"])

    # Same "not found" message regardless of whether the subject exists in
    # another investigation — don't leak the existence of cross-scope data.
    subject = await validate_subject(session, subject_id, investigation_id)
    if subject is None:
        return ToolResult(model_summary="scene_assembly: subject not found",
                          ui_payload={"events": []},
                          error="subject_not_in_scope")

    # Get the (subject, video) instance_id(s)
    si_rows = (await session.execute(
        select(SubjectInstance).where(
            SubjectInstance.subject_id == subject_id,
            SubjectInstance.video_id == video_id,
        )
    )).scalars().all()
    if not si_rows:
        return ToolResult(model_summary=f"scene_assembly({subject.label}): not in this video",
                          ui_payload={"events": [], "subject": subject.label})

    instance_ids = [r.instance_id for r in si_rows]

    # All detections for those instances, ordered by time
    dets = (await session.execute(
        select(Detection, Frame.timestamp_seconds, Frame.shot_id)
        .join(Frame, Frame.id == Detection.frame_id)
        .where(Detection.video_id == video_id, Detection.instance_id.in_(instance_ids))
        .order_by(Frame.timestamp_seconds)
    )).all()

    if not dets:
        return ToolResult(model_summary=f"scene_assembly({subject.label}): no detections",
                          ui_payload={"events": [], "subject": subject.label})

    # Group into runs of contiguous shots; classify enter/dwell/exit by run boundaries
    events = []
    cur = None
    for d, ts, shot_id in dets:
        if cur is None or shot_id != cur["shot_id"]:
            if cur is not None:
                events.append(cur)
            cur = {
                "shot_id": shot_id, "start_seconds": ts, "end_seconds": ts,
                "first_bbox": d.bbox, "last_bbox": d.bbox, "frame_count": 1,
            }
        else:
            cur["end_seconds"] = ts
            cur["last_bbox"] = d.bbox
            cur["frame_count"] += 1
    if cur is not None:
        events.append(cur)

    # Annotate each event: kind (enter/dwell/exit), nearest caption, nearby OCR
    for i, ev in enumerate(events):
        if i == 0 and len(events) > 1:
            ev["kind"] = "entered"
        elif i == len(events) - 1 and len(events) > 1:
            ev["kind"] = "exited"
        else:
            ev["kind"] = "dwell"

        # Nearest caption — same shot if any
        if ev["shot_id"]:
            cap = (await session.execute(
                select(Caption.text).where(Caption.shot_id == ev["shot_id"]).limit(1)
            )).scalar()
            if cap:
                ev["caption"] = cap

        # OCR within event window
        ocr_rows = (await session.execute(
            select(OcrText.text, OcrText.confidence)
            .join(Frame, Frame.id == OcrText.frame_id)
            .where(
                Frame.video_id == video_id,
                Frame.timestamp_seconds >= ev["start_seconds"],
                Frame.timestamp_seconds <= ev["end_seconds"],
            )
            .limit(5)
        )).all()
        if ocr_rows:
            ev["ocr"] = [{"text": r.text, "confidence": r.confidence} for r in ocr_rows]

        # Strip the shot_id UUID from the model-bound payload (UI keeps it)
        ev["shot_id"] = str(ev["shot_id"]) if ev["shot_id"] else None

    return ToolResult(
        model_summary=(
            f"scene_assembly({subject.label} in video): "
            f"{len(events)} events from {events[0]['start_seconds']:.1f}s "
            f"to {events[-1]['end_seconds']:.1f}s"
        ),
        ui_payload={"events": events, "subject": subject.label,
                    "video_id": str(video_id), "instance_ids": instance_ids},
        top_k_used=len(events),
    )
