"""Per-detection SigLIP box embeddings (the appearance signal that
underpins instance retrieval and image-vs-box subject matching).

Fake mode keys the embedding by (instance_id, video_id) so the same track
across many frames yields highly similar embeddings, while different tracks
are distinct — enough to verify the subject-matching logic end-to-end."""
import hashlib
import logging
from uuid import UUID

from sqlalchemy import select

from app.config import get_settings
from app.models import Detection
from app.worker.celery_app import celery_app
from app.worker.db import session_scope
from app.worker.util import deterministic_unit_vector, update_status

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(name="app.worker.tasks.embed_box.run", bind=True, acks_late=True)
def run(self, video_id: str) -> int:
    vid = UUID(video_id)
    with session_scope() as db:
        update_status(db, vid, stage="embed_box", status="processing")
        dets: list[Detection] = list(db.execute(
            select(Detection).where(Detection.video_id == vid)
        ).scalars().all())
        if not dets:
            return 0

        if settings.use_fake_ml:
            for d in dets:
                # Cluster around a per-(instance_id, video_id) anchor + small per-detection jitter
                key = f"{vid}:{d.instance_id}".encode()
                anchor = deterministic_unit_vector(key, dim=512)
                jitter = deterministic_unit_vector(d.id.bytes, dim=512)
                # 92% anchor / 8% jitter — same instance stays close, different instances diverge
                mixed = [0.92 * a + 0.08 * j for a, j in zip(anchor, jitter)]
                # renormalize
                norm = sum(x * x for x in mixed) ** 0.5
                d.box_embedding = [x / (norm + 1e-9) for x in mixed]
        else:
            _real_siglip_box(db, dets)

        db.commit()
        update_status(db, vid, progress_pct=65)
        return len(dets)


def _real_siglip_box(db, dets):
    """Crop each detection's bbox from its frame image and embed the crop
    with SigLIP. Frames are loaded once and shared across their detections."""
    from collections import defaultdict

    from PIL import Image
    from sqlalchemy import select

    from app.models import Frame
    from app.worker.ml import siglip_encode_images

    # Group detections by frame to avoid re-reading the same JPEG N times
    by_frame: dict = defaultdict(list)
    for d in dets:
        by_frame[d.frame_id].append(d)

    frames = (db.execute(
        select(Frame).where(Frame.id.in_(list(by_frame.keys())))
    ).scalars().all())
    frame_paths = {f.id: f.filepath for f in frames}

    crops: list = []
    crop_dets: list = []  # parallel list to crops
    load_failures = 0
    invalid_boxes = 0
    for fid, det_list in by_frame.items():
        path = frame_paths.get(fid)
        if not path:
            load_failures += 1
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            load_failures += 1
            logger.warning("embed_box: failed to open frame %s (%s)", path, e)
            continue
        w, h = img.size
        for d in det_list:
            b = d.bbox
            x1 = max(0, int(b["x1"] * w)); y1 = max(0, int(b["y1"] * h))
            x2 = min(w, int(b["x2"] * w)); y2 = min(h, int(b["y2"] * h))
            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
            crop_dets.append(d)

    if not crops:
        raise RuntimeError(
            f"embed_box: 0 valid crops from {len(dets)} detections "
            f"({load_failures} frame loads failed, {invalid_boxes} invalid bboxes) — "
            "instance retrieval will be broken"
        )
    if load_failures or invalid_boxes:
        logger.warning("embed_box: %d frame loads failed, %d invalid bboxes",
                       load_failures, invalid_boxes)

    embeddings = siglip_encode_images(crops, batch_size=32)
    for d, emb in zip(crop_dets, embeddings):
        d.box_embedding = [float(x) for x in emb]
    logger.info("[embed_box] embedded %d/%d detections", len(crop_dets), len(dets))


# Hash-helper kept local to avoid future import drift
def _key_hash(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()