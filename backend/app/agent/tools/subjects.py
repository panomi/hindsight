"""register_subject — turn confirmed detections into a tracked Subject.

The plan demands the N=20 farthest-point sampling cap, set in
settings.subject_reference_max. Also seeds subject_instances from the
source (video_id, instance_id) tuples of the confirmed detections.
"""
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.farthest_point import cap_reference_embeddings
from app.agent.schemas import ToolResult
from app.config import get_settings
from app.models import Detection, Subject, SubjectInstance, SubjectReference

settings = get_settings()

SCHEMA = {
    "name": "register_subject",
    "description": (
        "Create a new Subject from user-confirmed detections. Per the "
        "subject-registration-protocol, require ≥3 detections across varied "
        "frames/poses. The reference embedding set is capped at 20 via "
        "farthest-point sampling. "
        "IMPORTANT: confirmed_detection_ids MUST be Detection UUIDs (the "
        "'detection_id' field returned by detection-producing tools such as "
        "get_object_detections, open_vocab_detect, search_instances_by_subject, "
        "or search_visual_embeddings when they return per-detection items). "
        "DO NOT pass frame_ids, shot_ids, or caption_ids — those refer to "
        "whole frames and have no person-crop embeddings. If the user "
        "confirmed items by frame, first run get_object_detections on those "
        "frames to get the underlying detection_ids, then pass those here."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "description": "Stable handle, e.g. 'Subject A'"},
            "kind": {"type": "string", "enum": ["person", "vehicle", "object"]},
            "confirmed_detection_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "description": "Detection UUIDs only — not frame_ids.",
            },
        },
        "required": ["label", "kind", "confirmed_detection_ids"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    label = params["label"]
    kind = params["kind"]
    try:
        det_ids = [UUID(d) for d in params["confirmed_detection_ids"]]
    except (ValueError, TypeError) as exc:
        return ToolResult(
            model_summary=(
                f"register_subject: confirmed_detection_ids contains a non-UUID value "
                f"({exc}). Pass Detection UUIDs (from get_object_detections / "
                f"search_instances_by_subject / open_vocab_detect — not integer "
                f"indices and not frame_ids)."
            ),
            ui_payload={}, error="invalid_id_format",
        )
    if len(det_ids) < 3:
        return ToolResult(
            model_summary="register_subject: need ≥3 confirmed detections (skill: subject-registration-protocol)",
            ui_payload={}, error="too_few_detections",
        )

    dets = (await session.execute(
        select(Detection).where(Detection.id.in_(det_ids))
    )).scalars().all()

    # Diagnose precisely which failure mode we hit so the agent can self-correct.
    if len(dets) == 0:
        # IDs were valid UUIDs but matched zero rows in the detections table.
        # Almost always: the agent passed frame_ids (or shot_ids) thinking they
        # were detection_ids.  Suggest the right recovery path.
        return ToolResult(
            model_summary=(
                f"register_subject: 0 of {len(det_ids)} IDs matched any detection. "
                f"These are likely frame_ids, not detection_ids. To recover: call "
                f"get_object_detections with those frame timestamps (filter "
                f"class_name='person' for people) to get the underlying "
                f"detection_ids, then pass those here."
            ),
            ui_payload={}, error="ids_are_not_detections",
        )

    found_count = len(dets)
    missing = len(det_ids) - found_count
    embeddings = [list(d.box_embedding) for d in dets if d.box_embedding is not None]
    if len(embeddings) < 3:
        # We did find detections, but most lack box_embedding (rare — embed_box
        # stage should populate every detection).
        return ToolResult(
            model_summary=(
                f"register_subject: only {len(embeddings)}/{found_count} confirmed "
                f"detections have box_embeddings populated (need ≥3). "
                f"{missing} additional IDs did not match any detection row."
            ),
            ui_payload={}, error="insufficient_embeddings",
        )

    capped = cap_reference_embeddings(embeddings, settings.subject_reference_max)

    subject_id = uuid4()
    session.add(Subject(
        id=subject_id, investigation_id=investigation_id,
        label=label, kind=kind,
    ))
    for emb in capped:
        # ensure plain Python floats (cap_reference_embeddings uses numpy)
        session.add(SubjectReference(
            subject_id=subject_id, embedding=[float(x) for x in emb],
        ))

    pairs = {(d.video_id, d.instance_id) for d in dets if d.instance_id is not None}
    for v, iid in pairs:
        ins = insert(SubjectInstance).values(
            subject_id=subject_id, video_id=v, instance_id=iid,
            match_score=1.0, source="user_confirmed",
        ).on_conflict_do_nothing()
        await session.execute(ins)

    await session.commit()

    return ToolResult(
        model_summary=(
            f"register_subject('{label}', kind={kind}): "
            f"{len(capped)} reference embeddings (from {len(embeddings)} confirmed; "
            f"farthest-point capped to {settings.subject_reference_max}); "
            f"{len(pairs)} (video, instance) tuples seeded"
        ),
        ui_payload={
            "subject_id": str(subject_id), "label": label, "kind": kind,
            "reference_count": len(capped), "instance_pairs": len(pairs),
        },
    )
