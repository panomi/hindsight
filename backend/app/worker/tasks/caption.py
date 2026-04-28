"""Qwen2.5-VL captioning, scene-bounded (one caption per shot).

Real mode collects one representative frame per shot (TransNetV2 boundaries
guarantee visual distinctness), then runs Qwen2.5-VL-3B inference in batches
of 4 for throughput.  Fake mode synthesises captions from detected object
classes so caption_search works end-to-end without a GPU.
"""
import logging
from uuid import UUID, uuid4

from sqlalchemy import delete, select

from app.config import get_settings
from app.models import Caption, Detection, Frame, Shot
from app.worker.celery_app import celery_app
from app.worker.db import session_scope
from app.worker.util import clamp_text, deterministic_unit_vector, update_status

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(name="app.worker.tasks.caption.run", bind=True, acks_late=True)
def run(self, video_id: str) -> int:
    vid = UUID(video_id)
    with session_scope() as db:
        update_status(db, vid, stage="caption", status="processing")
        db.execute(delete(Caption).where(
            Caption.frame_id.in_(select(Frame.id).where(Frame.video_id == vid))
        ))

        shots: list[Shot] = list(db.execute(
            select(Shot).where(Shot.video_id == vid).order_by(Shot.shot_index)
        ).scalars().all())

        if not shots:
            raise RuntimeError(
                "caption: 0 shots found — shots stage must run before caption"
            )

        skipped_no_frame = 0
        empty_captions = 0

        # ── Collect one representative frame per shot ───────────────────────
        # TransNetV2 shot boundaries ensure each shot is visually distinct, so
        # we caption the earliest frame of each shot — no duplicate inference.
        pairs: list[tuple] = []  # (shot, frame)
        for s in shots:
            frame: Frame | None = db.execute(
                select(Frame).where(Frame.shot_id == s.id)
                .order_by(Frame.timestamp_seconds.asc())
            ).scalars().first()
            if frame is None:
                skipped_no_frame += 1
                continue
            pairs.append((s, frame))

        # ── Fake mode: synthesise captions from detections ──────────────────
        if settings.use_fake_ml:
            count = 0
            for s, frame in pairs:
                classes = [
                    d.class_name for d in db.execute(
                        select(Detection).where(Detection.frame_id == frame.id)
                    ).scalars().all()
                ]
                text = ("Scene contains " + ", ".join(sorted(set(classes))) + ".") if classes else "Empty scene."
                # 384-dim to match the new BGE-sized column.
                embedding = deterministic_unit_vector(text.encode() + frame.id.bytes, dim=384)
                db.add(Caption(id=uuid4(), shot_id=s.id, frame_id=frame.id,
                               text=text, source="fake", embedding=embedding))
                count += 1
            db.commit()

        # ── Real mode: batch Qwen inference ─────────────────────────────────
        else:
            from PIL import Image

            from app.worker.ml import bge_encode_text, qwen_vl_caption_batch

            images = [Image.open(f.filepath).convert("RGB") for _, f in pairs]
            raw_texts = qwen_vl_caption_batch(images)

            # Filter to non-empty captions before embedding so we batch-encode
            # everything in one BGE call (faster than per-row encoding).
            valid: list[tuple[Shot, Frame, str]] = []
            for (s, frame), text in zip(pairs, raw_texts, strict=True):
                if not text or not text.strip():
                    empty_captions += 1
                    logger.warning("caption: empty output for shot %s frame %s", s.id, frame.filepath)
                    continue
                valid.append((s, frame, clamp_text(text, source="caption")))

            count = 0
            if valid:
                texts = [t for _, _, t in valid]
                # Text-text retrieval — BGE-small replaces the previous
                # SigLIP-text path (mirrors transcripts, chunk 4).
                embeddings = bge_encode_text(texts)
                for (s, frame, text), emb in zip(valid, embeddings, strict=True):
                    db.add(Caption(
                        id=uuid4(), shot_id=s.id, frame_id=frame.id,
                        text=text, source="vlm",
                        embedding=[float(x) for x in emb],
                    ))
                    count += 1
            db.commit()
        update_status(db, vid, progress_pct=88)

        if count == 0:
            raise RuntimeError(
                f"caption: produced 0 captions from {len(shots)} shots "
                f"(skipped_no_frame={skipped_no_frame}, empty={empty_captions})"
            )
        if skipped_no_frame or empty_captions:
            logger.warning("[caption] %d/%d shots captioned (skipped_no_frame=%d, empty=%d)",
                           count, len(shots), skipped_no_frame, empty_captions)
        else:
            logger.info("[caption] produced %d captions for %d shots", count, len(shots))
        return count

