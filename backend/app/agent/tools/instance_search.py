"""Two-pass cross-video instance retrieval for a Subject.

Pass 1: For each reference embedding in the subject's set, do an ANN
        max-similarity match over detections.box_embedding.
Pass 2: For each matched detection, expand to all detections sharing
        (video_id, instance_id) — i.e. the rest of that within-video track.
The matched (video_id, instance_id) pairs are upserted to subject_instances
so future queries can join through them directly.
"""
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import ToolResult
from app.agent.tools._utils import empty_scope_result, scope_videos, validate_subject
from app.config import get_settings
from app.models import Detection, SubjectInstance, SubjectReference

settings = get_settings()

SCHEMA = {
    "name": "search_instances_by_subject",
    "description": "Find all detections matching a registered Subject across "
                   "videos. Two passes: (1) ANN over box embeddings vs the "
                   "subject's reference set; (2) expand within each matched "
                   "video by instance_id (within-video tracker continuity).",
    "input_schema": {
        "type": "object",
        "properties": {
            "subject_id": {"type": "string"},
            "top_k": {"type": "integer", "default": 30, "minimum": 1, "maximum": 50},
            "match_threshold": {"type": "number", "default": 0.6,
                                "description": "Minimum cosine similarity for pass-1 match."},
        },
        "required": ["subject_id"],
    },
}


async def run(session: AsyncSession, params: dict, investigation_id: UUID) -> ToolResult:
    subject_id = UUID(params["subject_id"])
    top_k = min(int(params.get("top_k", 30)), settings.tool_result_top_k_default)
    threshold = float(params.get("match_threshold", 0.6))

    # Subject must belong to this investigation; deny otherwise without
    # leaking whether the UUID exists elsewhere.
    subject = await validate_subject(session, subject_id, investigation_id)
    if subject is None:
        return ToolResult(
            model_summary=f"search_instances_by_subject({subject_id}): subject not found",
            ui_payload={"results": [], "subject_id": str(subject_id)},
            error="subject_not_in_scope",
        )

    # Restrict the ANN search to the investigation's collection so we don't
    # surface tracks from other investigations.
    video_ids = await scope_videos(session, investigation_id, [])
    if not video_ids:
        return empty_scope_result("search_instances_by_subject")

    refs = (await session.execute(
        select(SubjectReference.embedding).where(SubjectReference.subject_id == subject_id)
    )).scalars().all()
    if not refs:
        return ToolResult(
            model_summary=f"search_instances_by_subject({subject.label}): no reference embeddings",
            ui_payload={"results": [], "subject_id": str(subject_id)},
        )

    # Pass 1 — for each reference embedding, query the top-K nearest detections
    # WITHIN this investigation's videos.
    candidate_ids: dict[UUID, float] = {}  # detection_id -> best score
    for ref in refs:
        stmt = (
            select(
                Detection.id, Detection.video_id, Detection.instance_id,
                Detection.box_embedding.cosine_distance(ref).label("distance"),
            )
            .where(Detection.box_embedding.is_not(None))
            .where(Detection.video_id.in_(video_ids))
            .order_by("distance")
            .limit(top_k)
        )
        for row in (await session.execute(stmt)).all():
            score = float(1.0 - (row.distance or 1.0))
            if score >= threshold:
                if row.id not in candidate_ids or score > candidate_ids[row.id]:
                    candidate_ids[row.id] = score

    if not candidate_ids:
        return ToolResult(
            model_summary=f"search_instances_by_subject({subject.label}): 0 cross-video matches above {threshold}",
            ui_payload={"results": [], "subject_id": str(subject_id)},
        )

    # Pull metadata for matched detections; record (video_id, instance_id) pairs
    matched = (await session.execute(
        select(Detection).where(Detection.id.in_(candidate_ids.keys()))
    )).scalars().all()

    pairs: dict[tuple[UUID, int], float] = {}  # (video_id, instance_id) -> max score
    for d in matched:
        if d.instance_id is None:
            continue
        key = (d.video_id, d.instance_id)
        s = candidate_ids[d.id]
        if key not in pairs or s > pairs[key]:
            pairs[key] = s

    # Upsert subject_instances
    if pairs:
        rows = [
            {"subject_id": subject_id, "video_id": v, "instance_id": iid,
             "match_score": s, "source": "embedding_match"}
            for (v, iid), s in pairs.items()
        ]
        ins = insert(SubjectInstance).values(rows)
        ins = ins.on_conflict_do_update(
            index_elements=["subject_id", "video_id", "instance_id"],
            set_={"match_score": ins.excluded.match_score,
                  "source": ins.excluded.source},
        )
        await session.execute(ins)
        await session.commit()

    # Pass 2 — expand to all detections in the matched (video, instance) tuples
    if pairs:
        from sqlalchemy import tuple_
        expand_stmt = select(Detection).where(
            tuple_(Detection.video_id, Detection.instance_id).in_(list(pairs.keys()))
        ).limit(settings.tool_result_top_k_default)
        expanded = (await session.execute(expand_stmt)).scalars().all()
    else:
        expanded = matched

    items = [{
        "detection_id": str(d.id),
        "frame_id": str(d.frame_id),
        "video_id": str(d.video_id),
        "instance_id": d.instance_id,
        "class_name": d.class_name,
        "bbox": d.bbox,
        "score": candidate_ids.get(d.id, pairs.get((d.video_id, d.instance_id), 0.0)),
    } for d in expanded]

    return ToolResult(
        model_summary=(
            f"search_instances_by_subject({subject.label}): "
            f"{len(pairs)} matching tracks across {len({v for v, _ in pairs.keys()})} video(s); "
            f"expanded to {len(items)} detections."
        ),
        ui_payload={
            "results": items,
            "subject_id": str(subject_id),
            "matched_pairs": [{"video_id": str(v), "instance_id": iid, "score": s}
                              for (v, iid), s in pairs.items()],
        },
        top_k_used=len(items),
    )
