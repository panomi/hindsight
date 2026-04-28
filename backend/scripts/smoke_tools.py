"""Smoke-test agent tools against an already-ingested video.

Run after `alembic upgrade head` and at least one successful ingestion.
Picks the most recent ready video and walks through the key tool paths.
"""
import asyncio
from uuid import uuid4

from sqlalchemy import select

from app.agent.tools import (detection_query, instance_search, reranking,
                             scene_assembly, subjects, temporal_cluster,
                             visual_search)
from app.database import SessionLocal
from app.models import Detection, Investigation, Video


async def main():
    async with SessionLocal() as s:
        # Pick most recent ready video
        v: Video | None = (await s.execute(
            select(Video).where(Video.status == "ready").order_by(Video.created_at.desc()).limit(1)
        )).scalar_one_or_none()
        assert v is not None, "no ready video; run ingest first"
        print(f"video: {v.filename} id={v.id} status={v.status}")

        # Create an in-memory investigation row so subjects FK is satisfied
        inv = Investigation(id=uuid4(), collection_id=v.collection_id, title="smoke")
        s.add(inv); await s.commit(); await s.refresh(inv)

        # 1) get_object_detections(class=person)
        r = await detection_query.run(s, {"video_ids": [str(v.id)], "classes": ["person"], "min_confidence": 0.5}, inv.id)
        print(f"\n[1] {r.model_summary}")
        assert r.top_k_used >= 10, f"expected ≥10 person detections, got {r.top_k_used}"

        # 2) search_visual_embeddings
        r = await visual_search.run(s, {"query": "any person", "video_ids": [str(v.id)], "top_k": 5}, inv.id)
        print(f"[2] {r.model_summary}")
        assert r.top_k_used > 0

        # 3) register_subject from 3 person detections sharing instance_id=1
        det_ids = [str(d.id) for d in (await s.execute(
            select(Detection).where(Detection.video_id == v.id, Detection.instance_id == 1).limit(5)
        )).scalars().all()]
        assert len(det_ids) >= 3
        r = await subjects.run(s, {"label": "Subject A", "kind": "person",
                                   "confirmed_detection_ids": det_ids[:3]}, inv.id)
        print(f"[3] {r.model_summary}")
        subject_id = r.ui_payload["subject_id"]

        # 4) search_instances_by_subject — should expand via instance_id within the video
        r = await instance_search.run(s, {"subject_id": subject_id, "top_k": 30,
                                          "match_threshold": 0.5}, inv.id)
        print(f"[4] {r.model_summary}")
        assert r.top_k_used > 0, "expected expansion via instance_id"

        # 5) scene_assembly for that subject in that video
        r = await scene_assembly.run(s, {"subject_id": subject_id,
                                         "video_id": str(v.id)}, inv.id)
        print(f"[5] {r.model_summary}")

        # 6) temporal_cluster on the matched frames
        det_rows = (await s.execute(
            select(Detection.frame_id).where(Detection.video_id == v.id, Detection.instance_id == 1)
        )).scalars().all()
        r = await temporal_cluster.run(s, {"frame_ids": [str(f) for f in det_rows], "gap_sec": 3.0}, inv.id)
        print(f"[6] {r.model_summary}")

        # 7) apply_user_feedback — confirm 2 more detections, expect set growth
        more = [str(d.id) for d in (await s.execute(
            select(Detection).where(Detection.video_id == v.id, Detection.instance_id == 2).limit(2)
        )).scalars().all()]
        r = await reranking.run(s, {"subject_id": subject_id,
                                    "confirmed_detection_ids": more,
                                    "rejected_detection_ids": [],
                                    "propagate_by_instance": True}, inv.id)
        print(f"[7] {r.model_summary}")

        print("\n✅ all tool paths ok")


if __name__ == "__main__":
    asyncio.run(main())
