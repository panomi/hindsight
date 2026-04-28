"""Object detection (RT-DETR) + within-video tracking (ByteTrack).

Fake mode: synthesize one or two stable 'person' tracks across all frames
with consistent instance_ids — enough to exercise the instance-retrieval
path of the agent without real models.
"""
import logging
from uuid import UUID, uuid4

from sqlalchemy import delete, select

from app.config import get_settings
from app.models import Detection, Frame
from app.worker.celery_app import celery_app
from app.worker.db import session_scope
from app.worker.util import update_status

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(name="app.worker.tasks.detect_track.run", bind=True, acks_late=True)
def run(self, video_id: str) -> int:
    vid = UUID(video_id)
    with session_scope() as db:
        update_status(db, vid, stage="detect_track", status="processing")
        db.execute(delete(Detection).where(Detection.video_id == vid))

        frames: list[Frame] = list(db.execute(
            select(Frame).where(Frame.video_id == vid).order_by(Frame.timestamp_seconds)
        ).scalars().all())
        if not frames:
            return 0

        if settings.use_fake_ml:
            dets = _fake_detections(vid, frames)
        else:
            dets = _real_rtdetr_bytetrack(vid, frames)

        for d in dets:
            db.add(d)
        db.commit()
        update_status(db, vid, progress_pct=35)
        return len(dets)


def _fake_detections(vid: UUID, frames: list[Frame]) -> list[Detection]:
    """Two persistent person tracks: instance_id 1 (left) and 2 (right).
    Drift slightly per frame to look real."""
    import math
    out: list[Detection] = []
    for i, f in enumerate(frames):
        # Track 1 — left side, slow horizontal drift
        x1 = 0.10 + 0.02 * math.sin(i / 8.0)
        out.append(Detection(
            id=uuid4(), frame_id=f.id, video_id=vid,
            class_name="person", confidence=0.92,
            bbox={"x1": x1, "y1": 0.30, "x2": x1 + 0.18, "y2": 0.85},
            instance_id=1, box_embedding=None,  # filled by embed_box stage
        ))
        # Track 2 — right side, present every other frame
        if i % 2 == 0:
            x1 = 0.65 + 0.02 * math.cos(i / 8.0)
            out.append(Detection(
                id=uuid4(), frame_id=f.id, video_id=vid,
                class_name="person", confidence=0.88,
                bbox={"x1": x1, "y1": 0.32, "x2": x1 + 0.20, "y2": 0.88},
                instance_id=2, box_embedding=None,
            ))
        # Add an occasional vehicle detection
        if i % 5 == 0:
            out.append(Detection(
                id=uuid4(), frame_id=f.id, video_id=vid,
                class_name="car", confidence=0.81,
                bbox={"x1": 0.40, "y1": 0.55, "x2": 0.60, "y2": 0.80},
                instance_id=10, box_embedding=None,
            ))
    return out


def _real_rtdetr_bytetrack(vid: UUID, frames: list[Frame]) -> list[Detection]:
    """Per-frame RT-DETR detection, fed into ByteTrack to assign instance_ids
    that persist across frames within this video."""
    from PIL import Image

    from app.worker.ml import bytetrack_new_tracker, bytetrack_step, rtdetr_detect

    out: list[Detection] = []
    if not frames:
        return out

    # Load all frame images (capped working-set — 1 fps × ~10min == 600 imgs ≈ 1 GB RAM)
    images: list[Image.Image] = []
    load_failures = 0
    for f in frames:
        try:
            images.append(Image.open(f.filepath).convert("RGB"))
        except Exception as e:
            images.append(None)  # type: ignore
            load_failures += 1
            logger.warning("detect_track: failed to open frame %s (%s)", f.filepath, e)

    keep_idx = [i for i, img in enumerate(images) if img is not None]
    valid_imgs = [images[i] for i in keep_idx]
    if not valid_imgs:
        raise RuntimeError(
            f"detect_track: 0/{len(frames)} frames could load — check frames stage output"
        )
    if load_failures:
        logger.warning("detect_track: %d/%d frames failed to load — running on remainder",
                       load_failures, len(frames))

    detections_per_frame = rtdetr_detect(valid_imgs, threshold=0.4)
    # Reinflate to per-input-frame, with [] for failed loads
    full_dets: list[list[dict]] = [[] for _ in frames]
    for k, idx in enumerate(keep_idx):
        full_dets[idx] = detections_per_frame[k]

    # Track within video
    import numpy as np
    tracker = bytetrack_new_tracker()
    for f, img, dets in zip(frames, images, full_dets):
        if img is None:
            continue
        tracked = bytetrack_step(tracker, dets, img.width, img.height,
                                 img_numpy=np.array(img))
        for d in tracked:
            out.append(Detection(
                id=uuid4(), frame_id=f.id, video_id=vid,
                class_name=d["class_name"], confidence=float(d["confidence"]),
                bbox=d["bbox"], instance_id=d.get("instance_id"),
                box_embedding=None,  # filled by embed_box
            ))

    n_instances = len({d.instance_id for d in out if d.instance_id is not None})
    logger.info("[detect_track] %d frames → %d detections, %d unique tracks",
                len(frames) - load_failures, len(out), n_instances)
    return out
