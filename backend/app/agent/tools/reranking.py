"""apply_user_feedback — incorporate confirm/reject signals into a Subject.

Confirms expand the reference set (then prune via farthest-point cap).
Rejects don't poison the set; instead they are flagged so the
re-ranking-protocol skill can advise whether to spawn a sibling subject.
"""
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.farthest_point import cap_reference_embeddings
from app.agent.schemas import ToolResult
from app.agent.tools._utils import validate_subject
from app.config import get_settings
from app.models import Detection, SubjectInstance, SubjectReference

settings = get_settings()

SCHEMA = {
    "name": "apply_user_feedback",
    "description": "Update a Subject from user confirm/reject feedback. Confirms "
                   "expand the reference embedding set (capped via farthest-point "
                   "sampling). Rejects are recorded but never added to the set. "
                   "If propagate_by_instance=true, confirmed (video_id, "
                   "instance_id) pairs are added to subject_instances so future "
                   "queries pick up all detections in the matched track.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subject_id": {"type": "string"},
            "confirmed_detection_ids": {"type": "array", "items": {"type": "string"}, "default": []},
            "rejected_detection_ids": {"type": "array", "items": {"type": "string"}, "default": []},
            "propagate_by_instance": {"type": "boolean", "default": True},
        },
        "required": ["subject_id"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    subject_id = UUID(params["subject_id"])
    confirmed_ids = [UUID(d) for d in params.get("confirmed_detection_ids", [])]
    rejected_ids = [UUID(d) for d in params.get("rejected_detection_ids", [])]
    propagate = bool(params.get("propagate_by_instance", True))

    subject = await validate_subject(session, subject_id, investigation_id)
    if subject is None:
        return ToolResult(
            model_summary=f"apply_user_feedback: subject {subject_id} not found",
            ui_payload={}, error="subject_not_in_scope",
        )

    # Confirmed detections — add their box embeddings to the set, then cap
    if confirmed_ids:
        new = (await session.execute(
            select(Detection).where(Detection.id.in_(confirmed_ids))
        )).scalars().all()
        new_embs = [list(d.box_embedding) for d in new if d.box_embedding is not None]
        existing = [list(r.embedding) for r in (await session.execute(
            select(SubjectReference).where(SubjectReference.subject_id == subject_id)
        )).scalars().all()]
        merged = existing + new_embs
        capped = cap_reference_embeddings(merged, settings.subject_reference_max)
        # Replace the set
        await session.execute(
            delete(SubjectReference).where(SubjectReference.subject_id == subject_id)
        )
        for emb in capped:
            session.add(SubjectReference(
                subject_id=subject_id, embedding=[float(x) for x in emb],
            ))

        if propagate:
            pairs = {(d.video_id, d.instance_id) for d in new if d.instance_id is not None}
            for v, iid in pairs:
                ins = insert(SubjectInstance).values(
                    subject_id=subject_id, video_id=v, instance_id=iid,
                    match_score=1.0, source="user_confirmed",
                )
                ins = ins.on_conflict_do_update(
                    index_elements=["subject_id", "video_id", "instance_id"],
                    set_={"match_score": 1.0, "source": "user_confirmed"},
                )
                await session.execute(ins)
    else:
        capped = [list(r.embedding) for r in (await session.execute(
            select(SubjectReference).where(SubjectReference.subject_id == subject_id)
        )).scalars().all()]

    # Rejection clustering hint: if many rejects share an embedding cluster,
    # the agent should consider spawning a sibling subject (re-ranking-protocol).
    reject_meta = []
    if rejected_ids:
        rej = (await session.execute(
            select(Detection).where(Detection.id.in_(rejected_ids))
        )).scalars().all()
        reject_meta = [{
            "detection_id": str(d.id),
            "video_id": str(d.video_id),
            "instance_id": d.instance_id,
        } for d in rej]

    await session.commit()

    return ToolResult(
        model_summary=(
            f"apply_user_feedback({subject.label}): "
            f"+{len(confirmed_ids)} confirmed (set now {len(capped)}/"
            f"{settings.subject_reference_max}), "
            f"-{len(rejected_ids)} rejected"
        ),
        ui_payload={
            "subject_id": str(subject_id),
            "reference_count": len(capped),
            "rejected": reject_meta,
        },
    )
